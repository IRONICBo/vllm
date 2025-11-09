# FireRedASR-L 模型集成文档

本文档详细说明了 FireRedASR-L 模型在 vLLM 中的集成实现。

## 目录

1. [概述](#概述)
2. [架构说明](#架构说明)
3. [代码实现详解](#代码实现详解)
4. [使用方法](#使用方法)
5. [测试](#测试)

## 概述

FireRedASR 是一个基于注意力机制的编码器-解码器（Attention-based Encoder-Decoder, AED）语音识别模型，专门为中文和英文语音识别设计。

### 主要特性

- **编码器架构**: Conformer（结合了 Transformer 和卷积网络的优势）
- **解码器架构**: Transformer
- **支持语言**: 中文（zh）和英文（en）
- **音频输入**: 16kHz, 16-bit PCM 格式
- **最大音频长度**: 60秒（FireRedASR-AED）
- **分布式支持**: 完全支持 Tensor Parallel

## 架构说明

### 整体架构

```
音频输入 (16kHz PCM)
    ↓
特征提取 (80-dim Mel-spectrogram)
    ↓
Conformer Encoder (12层)
    ↓
Transformer Decoder (6层)
    ↓
输出文本
```

### Conformer Encoder

Conformer 是一种混合架构，结合了以下组件：

1. **Feed-Forward Module (FFN1)**: 半步残差连接
2. **Multi-Head Self-Attention**: 捕获长距离依赖
3. **Convolution Module**: 捕获局部特征
4. **Feed-Forward Module (FFN2)**: 半步残差连接

每个 Conformer Block 的计算流程：

```python
x = x + 0.5 * FFN1(x)      # 半步前馈
x = x + MHSA(x)             # 多头自注意力
x = x + Conv(x)             # 卷积模块
x = x + 0.5 * FFN2(x)      # 半步前馈
```

### Convolution Module

卷积模块包含：

1. **Pointwise Expansion**: 将特征维度扩展到 2x
2. **GLU Activation**: 门控线性单元激活
3. **Depthwise Convolution**: 深度可分离卷积（kernel_size=31）
4. **Batch Normalization**: 批归一化
5. **Swish Activation**: Swish 激活函数
6. **Pointwise Compression**: 将特征维度压缩回原始大小

### Transformer Decoder

标准 Transformer 解码器，包含：

1. **Masked Self-Attention**: 掩码自注意力（因果掩码）
2. **Cross-Attention**: 与编码器输出的交叉注意力
3. **Feed-Forward Network**: 前馈网络

## 代码实现详解

### 1. 模型配置 (`FireRedASRConfig`)

位置: `vllm/transformers_utils/configs/fireredasr.py`

```python
class FireRedASRConfig(PretrainedConfig):
    model_type = "fireredasr_aed"

    def __init__(
        self,
        vocab_size: int = 8000,        # 词汇表大小
        d_model: int = 512,             # 模型维度
        n_layers_enc: int = 12,         # 编码器层数
        n_layers_dec: int = 6,          # 解码器层数
        n_head: int = 8,                # 注意力头数
        kernel_size: int = 31,          # 卷积核大小
        dropout_rate: float = 0.1,      # Dropout 率
        residual_dropout: float = 0.1,  # 残差 Dropout
        pe_maxlen: int = 5000,          # 位置编码最大长度
        idim: int = 80,                 # 输入特征维度（Mel bins）
        odim: int = 8000,               # 输出维度
        sos_id: int = 1,                # 开始标记 ID
        eos_id: int = 2,                # 结束标记 ID
        pad_id: int = 0,                # 填充标记 ID
        **kwargs
    ):
        # 初始化配置...
```

### 2. 关键组件实现

#### FireRedASRConformerBlock

位置: `vllm/model_executor/models/fireredasr_aed.py`

```python
class FireRedASRConformerBlock(nn.Module):
    """Conformer block combining self-attention, convolution, and feed-forward."""

    def __init__(self, d_model, n_head, d_ff, kernel_size, dropout, ...):
        # 初始化四个主要模块
        self.ffn1 = FireRedASRFeedForward(...)
        self.self_attn = FireRedASRAttention(...)
        self.conv = FireRedASRConvModule(...)
        self.ffn2 = FireRedASRFeedForward(...)

    def forward(self, x, attn_mask=None):
        # Conformer 的计算流程
        x = x + 0.5 * self.dropout(self.ffn1(self.norm_ffn1(x)))
        x = x + self.dropout(self.self_attn(self.norm_attn(x), attn_mask))
        x = x + self.dropout(self.conv(self.norm_conv(x)))
        x = x + 0.5 * self.dropout(self.ffn2(self.norm_ffn2(x)))
        return self.norm_final(x)
```

#### FireRedASRConvModule

```python
class FireRedASRConvModule(nn.Module):
    """Convolution module for Conformer block."""

    def forward(self, x):
        # 1. Pointwise expansion (1x1 conv)
        x = self.pointwise_conv1(x)

        # 2. GLU activation
        x = self.glu(x)

        # 3. Depthwise convolution
        x = x.transpose(1, 2)  # (batch, seq, channels) -> (batch, channels, seq)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)  # 转回

        # 4. Swish activation
        x = self.activation(x)

        # 5. Pointwise compression
        x = self.pointwise_conv2(x)
        return self.dropout(x)
```

### 3. MultiModal 处理器

#### FireRedASRMultiModalProcessor

位置: `vllm/model_executor/models/fireredasr_aed.py`

这个处理器负责处理音频输入数据：

```python
class FireRedASRMultiModalProcessor(EncDecMultiModalProcessor):
    def _get_data_parser(self):
        # 返回音频解析器，目标采样率 16kHz
        return MultiModalDataParser(target_sr=16000)

    def _call_hf_processor(self, prompt, mm_data, ...):
        # 处理音频数据，转换为模型输入特征
        if "audio" in mm_data:
            # 将音频转换为 mel-spectrogram 特征
            # 输出形状: (batch, time_steps, 80)
            ...
```

### 4. 模型主类

#### FireRedASRForConditionalGeneration

```python
@MULTIMODAL_REGISTRY.register_processor(
    FireRedASRMultiModalProcessor,
    info=FireRedASRProcessingInfo,
    dummy_inputs=FireRedASRDummyInputsBuilder
)
class FireRedASRForConditionalGeneration(nn.Module, SupportsTranscription, SupportsMultiModal):
    # 实现 SupportsTranscription 接口
    supports_transcription_only = True
    supported_languages = FIREREDASR_SUPPORTED_LANGS

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        # 创建编码器-解码器模型
        self.model = FireRedASRModel(vllm_config=vllm_config, prefix=prefix)

        # 输出投影层
        self.proj_out = ParallelLMHead(...)

        # Logits 处理器
        self.logits_processor = LogitsProcessor(...)

    def forward(self, input_ids, positions, **kwargs):
        # 1. 解析音频输入
        audio_input = self._parse_and_validate_audio_input(**kwargs)

        # 2. 编码器处理音频特征
        encoder_outputs = self.model.get_encoder_outputs(
            audio_input["input_features"],
            audio_input.get("input_lengths")
        )

        # 3. 解码器生成文本
        hidden_states = self.model.decoder(
            input_ids,
            encoder_outputs[0],  # encoder output
            encoder_outputs[2]   # encoder mask
        )

        return hidden_states
```

### 5. 模型注册

模型在两个地方注册：

1. **模型注册表**: `vllm/model_executor/models/registry.py`

```python
_MULTIMODAL_MODELS = {
    ...
    "FireRedASRForConditionalGeneration": (
        "fireredasr_aed",
        "FireRedASRForConditionalGeneration"
    ),
    ...
}
```

2. **配置注册**: `vllm/transformers_utils/configs/__init__.py`

```python
from vllm.transformers_utils.configs.fireredasr import FireRedASRConfig

__all__ = [
    ...
    "FireRedASRConfig",
    ...
]
```

## 使用方法

### 1. 基本推理

```python
from vllm import LLM

# 加载模型
llm = LLM(
    model="path/to/fireredasr-aed-l",
    task="transcribe",  # 或 "translate"
    dtype="float16",
    tensor_parallel_size=1,
)

# 准备音频（16kHz, 16-bit PCM）
import numpy as np

# 从文件加载音频
audio_data = load_audio("audio.wav")  # 返回 numpy array

# 生成转录
outputs = llm.generate({
    "prompt": "",
    "multi_modal_data": {
        "audio": audio_data
    }
})

print(outputs[0].outputs[0].text)
```

### 2. 使用 OpenAI 兼容 API

```python
from vllm import LLM
from vllm.entrypoints.openai.serving_transcription import OpenAIServingTranscription

# 启动服务器
llm = LLM(model="path/to/fireredasr-aed-l")

# 通过 HTTP 发送请求
import requests

response = requests.post(
    "http://localhost:8000/v1/audio/transcriptions",
    files={"file": open("audio.wav", "rb")},
    data={
        "model": "fireredasr-aed-l",
        "language": "zh",  # 或 "en"
    }
)

print(response.json()["text"])
```

### 3. 批量处理

```python
# 批量处理多个音频文件
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
audio_data = [load_audio(f) for f in audio_files]

outputs = llm.generate([
    {
        "prompt": "",
        "multi_modal_data": {"audio": audio}
    }
    for audio in audio_data
])

for output in outputs:
    print(output.outputs[0].text)
```

### 4. 音频预处理

使用 FFmpeg 将音频转换为正确格式：

```bash
# 转换为 16kHz, 单声道, 16-bit PCM
ffmpeg -i input.mp3 -ar 16000 -ac 1 -acodec pcm_s16le output.wav
```

Python 中的预处理：

```python
import librosa
import numpy as np

def load_audio(audio_path, target_sr=16000):
    """加载音频文件并转换为正确格式"""
    # 使用 librosa 加载音频
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    # 转换为 16-bit PCM 格式
    audio = (audio * 32767).astype(np.int16)

    return audio

# 使用
audio = load_audio("audio.wav")
```

## 测试

### 运行单元测试

```bash
# 运行所有 FireRedASR 测试
pytest tests/test_fireredasr_integration.py -v

# 运行特定测试
pytest tests/test_fireredasr_integration.py::TestFireRedASRConfig -v
pytest tests/test_fireredasr_integration.py::TestFireRedASRModel -v
pytest tests/test_fireredasr_integration.py::TestSupportsTranscriptionInterface -v
```

### 测试内容

1. **配置测试**: 验证 `FireRedASRConfig` 的默认值和自定义配置
2. **模型组件测试**: 测试 Conformer Encoder 和 Transformer Decoder 的初始化
3. **接口测试**: 验证 `SupportsTranscription` 接口的实现
4. **MultiModal 处理器测试**: 测试音频数据的处理流程
5. **模型注册测试**: 确认模型正确注册到 vLLM

### 集成测试示例

```python
def test_end_to_end_transcription():
    """端到端转录测试"""
    from vllm import LLM
    import numpy as np

    # 创建测试音频（1秒的正弦波）
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440Hz 正弦波
    audio = (audio * 32767).astype(np.int16)

    # 加载模型
    llm = LLM(model="path/to/fireredasr-aed-l")

    # 执行转录
    outputs = llm.generate({
        "prompt": "",
        "multi_modal_data": {"audio": audio}
    })

    # 验证输出
    assert len(outputs) > 0
    assert len(outputs[0].outputs[0].text) > 0
```

## 与 Whisper 的对比

| 特性 | FireRedASR | Whisper |
|------|------------|---------|
| 编码器 | Conformer (12层) | Transformer (24层) |
| 解码器 | Transformer (6层) | Transformer (24层) |
| 支持语言 | 中文、英文 | 98种语言 |
| 最大音频长度 | 60秒 | 30秒（分块处理） |
| 特征维度 | 80-dim Mel | 80-dim Mel |
| 采样率 | 16kHz | 16kHz |
| 参数量 | ~100M (L版本) | ~1.5B (large-v3) |

## 性能优化建议

1. **使用混合精度**: 设置 `dtype="float16"` 或 `dtype="bfloat16"`
2. **Tensor Parallel**: 对于大模型，使用 `tensor_parallel_size > 1`
3. **批处理**: 使用批量推理提高吞吐量
4. **音频预处理缓存**: 预先处理音频特征并缓存
5. **KV Cache**: 自动启用解码器 KV Cache

## 故障排除

### 问题 1: 位置编码错误

**错误**: `IndexError: index out of range`

**原因**: 音频长度超过了 `pe_maxlen` (5000)

**解决**:
- 将长音频分割为 60秒以内的片段
- 或增加配置中的 `pe_maxlen` 参数

### 问题 2: 音频格式错误

**错误**: `ValueError: Incorrect type of audio features`

**原因**: 音频格式不正确

**解决**:
```python
# 确保音频是正确的格式
audio = librosa.load(path, sr=16000, mono=True)[0]
audio = (audio * 32767).astype(np.int16)
```

### 问题 3: 语言不支持

**错误**: `ValueError: Language 'xx' not supported`

**原因**: 指定的语言不在支持列表中

**解决**: 仅使用 "zh" 或 "en"

## 总结

FireRedASR-L 在 vLLM 中的集成遵循了与 Whisper 相同的设计模式：

1. **编码器-解码器架构**: 音频编码器 + 文本解码器
2. **MultiModal 支持**: 通过 `MULTIMODAL_REGISTRY` 注册
3. **SupportsTranscription 接口**: 实现标准转录接口
4. **分布式支持**: 完整的 Tensor Parallel 支持
5. **OpenAI 兼容**: 可以使用 OpenAI API 格式

主要区别在于：
- Conformer 编码器（而非纯 Transformer）
- 仅支持中英文
- 更轻量级（参数量更小）
- 专注于中文识别性能

## 参考资料

- [FireRedASR GitHub](https://github.com/FireRedTeam/FireRedASR)
- [vLLM Documentation](https://docs.vllm.ai)
- [Conformer Paper](https://arxiv.org/abs/2005.08100)
- [Whisper Paper](https://arxiv.org/abs/2212.04356)
