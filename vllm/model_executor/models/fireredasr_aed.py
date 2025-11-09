# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal, Optional, Union, cast

import numpy as np
import torch
from torch import nn
from transformers import BatchFeature

from vllm.attention import Attention, AttentionType
from vllm.attention.layer import MultiHeadAttention
from vllm.attention.layers.cross_attention import CrossAttention
from vllm.config import (CacheConfig, ModelConfig, SpeechToTextConfig,
                         VllmConfig)
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, NestedTensors
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargsItems)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (BaseProcessingInfo,
                                        EncDecMultiModalProcessor,
                                        PromptReplacement, PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.transformers_utils.processor import cached_get_processor
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from vllm.transformers_utils.configs.fireredasr import FireRedASRConfig

from .interfaces import (MultiModalEmbeddings, SupportsMultiModal,
                         SupportsTranscription)
from .utils import (AutoWeightsLoader, WeightsMapper, cast_overflow_tensors,
                    make_layers)

logger = init_logger(__name__)

# FireRedASR supports Chinese and English
FIREREDASR_SUPPORTED_LANGS = {
    "zh": "Chinese",
    "en": "English",
}


class FireRedASRAudioInputs(TensorSchema):
    """
    Dimensions:
        - b: Batch size
        - t: Time frames
        - f: Feature dimension
    """
    input_features: Annotated[Optional[NestedTensors],
                              TensorShape("b", "t", "f")]


class FireRedASRPositionalEncoding(nn.Module):
    """Positional encoding for FireRedASR."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class FireRedASRAttention(nn.Module):
    """Multi-head attention for FireRedASR."""
    
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float = 0.1,
        attn_type: AttentionType = AttentionType.DECODER,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.attn_type = attn_type
        
        if attn_type == AttentionType.ENCODER:
            self.qkv_proj = QKVParallelLinear(
                d_model, self.head_dim, n_head,
                bias=True, quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj"
            )
        else:
            self.q_proj = ColumnParallelLinear(
                d_model, self.head_dim * n_head,
                bias=True, quant_config=quant_config,
                prefix=f"{prefix}.q_proj"
            )
            self.k_proj = ColumnParallelLinear(
                d_model, self.head_dim * n_head,
                bias=True, quant_config=quant_config,
                prefix=f"{prefix}.k_proj"
            )
            self.v_proj = ColumnParallelLinear(
                d_model, self.head_dim * n_head,
                bias=True, quant_config=quant_config,
                prefix=f"{prefix}.v_proj"
            )
            
        self.out_proj = RowParallelLinear(
            self.head_dim * n_head, d_model,
            bias=True, quant_config=quant_config,
            prefix=f"{prefix}.out_proj"
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape
        
        if self.attn_type == AttentionType.ENCODER:
            qkv, _ = self.qkv_proj(query)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            q, _ = self.q_proj(query)
            if key is None:
                key = query
            if value is None:
                value = key
            k, _ = self.k_proj(key)
            v, _ = self.v_proj(value)
            
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_head * self.head_dim)
        
        output, _ = self.out_proj(attn_output)
        return output


class FireRedASRFeedForward(nn.Module):
    """Feed-forward network for FireRedASR."""
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.linear1 = ColumnParallelLinear(
            d_model, d_ff,
            bias=True, quant_config=quant_config,
            prefix=f"{prefix}.linear1"
        )
        self.linear2 = RowParallelLinear(
            d_ff, d_model,
            bias=True, quant_config=quant_config,
            prefix=f"{prefix}.linear2"
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = get_act_fn(activation)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x, _ = self.linear2(x)
        return x


class FireRedASRConformerBlock(nn.Module):
    """Conformer block combining self-attention, convolution, and feed-forward."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: int,
        kernel_size: int,
        dropout: float = 0.1,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = d_model

        # Feed-forward module 1 (half-step)
        self.ffn1 = FireRedASRFeedForward(
            d_model, d_ff, dropout, "swish",
            quant_config=quant_config, prefix=f"{prefix}.ffn1"
        )

        # Multi-head self-attention
        self.self_attn = FireRedASRAttention(
            d_model, n_head, dropout, AttentionType.ENCODER,
            quant_config=quant_config, prefix=f"{prefix}.self_attn"
        )

        # Convolution module
        self.conv = FireRedASRConvModule(
            d_model, kernel_size, dropout,
            quant_config=quant_config, prefix=f"{prefix}.conv"
        )

        # Feed-forward module 2 (half-step)
        self.ffn2 = FireRedASRFeedForward(
            d_model, d_ff, dropout, "swish",
            quant_config=quant_config, prefix=f"{prefix}.ffn2"
        )

        self.norm_ffn1 = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_conv = nn.LayerNorm(d_model)
        self.norm_ffn2 = nn.LayerNorm(d_model)
        self.norm_final = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Feed-forward module 1 (half-step residual)
        residual = x
        x = self.norm_ffn1(x)
        x = residual + 0.5 * self.dropout(self.ffn1(x))

        # Multi-head self-attention
        residual = x
        x = self.norm_attn(x)
        x = residual + self.dropout(self.self_attn(x, attn_mask=attn_mask))

        # Convolution module
        residual = x
        x = self.norm_conv(x)
        x = residual + self.dropout(self.conv(x))

        # Feed-forward module 2 (half-step residual)
        residual = x
        x = self.norm_ffn2(x)
        x = residual + 0.5 * self.dropout(self.ffn2(x))

        # Final layer norm
        x = self.norm_final(x)

        return x


class FireRedASRConvModule(nn.Module):
    """Convolution module for Conformer block."""

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dropout: float = 0.1,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        # Pointwise expansion
        self.pointwise_conv1 = ColumnParallelLinear(
            d_model, d_model * 2,
            bias=True, quant_config=quant_config,
            prefix=f"{prefix}.pointwise_conv1"
        )

        # GLU activation
        self.glu = nn.GLU(dim=-1)

        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model,  # Depthwise
        )

        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = get_act_fn("swish")

        # Pointwise compression
        self.pointwise_conv2 = RowParallelLinear(
            d_model, d_model,
            bias=True, quant_config=quant_config,
            prefix=f"{prefix}.pointwise_conv2"
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pointwise expansion
        x, _ = self.pointwise_conv1(x)

        # GLU activation
        x = self.glu(x)

        # Depthwise convolution
        # x shape: (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)  # Back to (batch, seq_len, channels)

        x = self.activation(x)

        # Pointwise compression
        x, _ = self.pointwise_conv2(x)
        x = self.dropout(x)

        return x


class FireRedASRConformerEncoder(nn.Module):
    """Conformer encoder for FireRedASR."""

    def __init__(
        self,
        idim: int,
        d_model: int,
        n_layers: int,
        n_head: int,
        d_ff: int,
        kernel_size: int,
        dropout: float = 0.1,
        pe_maxlen: int = 5000,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = ColumnParallelLinear(
            idim, d_model,
            bias=True, quant_config=quant_config,
            prefix=f"{prefix}.input_proj"
        )

        # Positional encoding
        self.pos_enc = FireRedASRPositionalEncoding(d_model, pe_maxlen)

        # Conformer blocks
        self.layers = nn.ModuleList([
            FireRedASRConformerBlock(
                d_model, n_head, d_ff, kernel_size, dropout,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{i}"
            ) for i in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Input projection
        x, _ = self.input_proj(x)
        
        # Positional encoding
        x = self.pos_enc(x)
        x = self.dropout(x)
        
        # Create attention mask from input lengths
        batch_size, max_len = x.size(0), x.size(1)
        if input_lengths is not None:
            attn_mask = torch.arange(max_len, device=x.device).expand(
                batch_size, max_len) < input_lengths.unsqueeze(1)
        else:
            attn_mask = torch.ones(batch_size, max_len, device=x.device, dtype=torch.bool)
        
        # Apply conformer blocks
        for layer in self.layers:
            x = layer(x, attn_mask)
            
        x = self.norm(x)
        
        return x, input_lengths, attn_mask


class FireRedASRTransformerDecoderLayer(nn.Module):
    """Transformer decoder layer for FireRedASR."""
    
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: int,
        dropout: float = 0.1,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.self_attn = FireRedASRAttention(
            d_model, n_head, dropout, AttentionType.DECODER,
            cache_config=cache_config, quant_config=quant_config,
            prefix=f"{prefix}.self_attn"
        )
        self.cross_attn = FireRedASRAttention(
            d_model, n_head, dropout, AttentionType.DECODER,
            cache_config=cache_config, quant_config=quant_config,
            prefix=f"{prefix}.cross_attn"
        )
        self.feed_forward = FireRedASRFeedForward(
            d_model, d_ff, dropout, "relu",
            quant_config=quant_config, prefix=f"{prefix}.feed_forward"
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, attn_mask=self_attn_mask)
        x = self.dropout(x) + residual
        
        # Cross-attention
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, key=encoder_output, value=encoder_output,
                           attn_mask=cross_attn_mask)
        x = self.dropout(x) + residual
        
        # Feed-forward
        residual = x
        x = self.norm3(x)
        x = self.feed_forward(x)
        x = self.dropout(x) + residual
        
        return x


class FireRedASRTransformerDecoder(nn.Module):
    """Transformer decoder for FireRedASR."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_head: int,
        d_ff: int,
        dropout: float = 0.1,
        pe_maxlen: int = 5000,
        sos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        
        # Token embeddings
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size, d_model,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens"
        )
        
        # Positional encoding
        self.pos_enc = FireRedASRPositionalEncoding(d_model, pe_maxlen)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            FireRedASRTransformerDecoderLayer(
                d_model, n_head, d_ff, dropout,
                cache_config=cache_config, quant_config=quant_config,
                prefix=f"{prefix}.layers.{i}"
            ) for i in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Token embeddings
        x = self.embed_tokens(input_ids)
        x = x * math.sqrt(self.d_model)  # Scale embeddings
        
        # Positional encoding
        x = self.pos_enc(x)
        x = self.dropout(x)
        
        # Create causal mask for self-attention
        seq_len = input_ids.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        
        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, causal_mask, encoder_mask)
            
        x = self.norm(x)
        return x


class FireRedASRModel(nn.Module):
    """FireRedASR encoder-decoder model."""
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        
        self.encoder = FireRedASRConformerEncoder(
            idim=config.idim,
            d_model=config.d_model,
            n_layers=config.n_layers_enc,
            n_head=config.n_head,
            d_ff=config.d_model * 4,  # Standard transformer ratio
            kernel_size=config.kernel_size,
            dropout=config.dropout_rate,
            pe_maxlen=config.pe_maxlen,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.encoder"
        )
        
        self.decoder = FireRedASRTransformerDecoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers_dec,
            n_head=config.n_head,
            d_ff=config.d_model * 4,
            dropout=config.residual_dropout,
            pe_maxlen=config.pe_maxlen,
            sos_id=config.sos_id,
            eos_id=config.eos_id,
            pad_id=config.pad_id,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.decoder"
        )
        
    def forward(
        self,
        input_features: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor],
        input_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_outputs = self.get_encoder_outputs(input_features, input_lengths)
        if encoder_outputs is None:
            raise ValueError("Encoder outputs cannot be None")
            
        encoder_output, _, encoder_mask = encoder_outputs
        return self.decoder(input_ids, encoder_output, encoder_mask)
        
    def get_encoder_outputs(
        self,
        input_features: Optional[torch.Tensor],
        input_lengths: Optional[torch.Tensor] = None,
    ) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]]:
        if input_features is None:
            return None
        return self.encoder(input_features, input_lengths)


class FireRedASRProcessingInfo(BaseProcessingInfo):
    """Processing information for FireRedASR."""
    
    def get_hf_config(self) -> FireRedASRConfig:
        return self.ctx.get_hf_config(FireRedASRConfig)
        
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": 1}
        
    def get_num_audio_tokens(self) -> int:
        # Estimate based on typical audio processing
        return 3000  # Configurable based on model requirements


class FireRedASRDummyInputsBuilder(BaseDummyInputsBuilder[FireRedASRProcessingInfo]):
    """Dummy inputs builder for FireRedASR."""
    
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return "<|startoftranscript|>" * num_audios
        
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        if num_audios == 0:
            return {}
            
        # Create dummy audio features (batch_size=1, time_steps, feature_dim)
        dummy_audio = np.random.randn(1, 1000, 80).astype(np.float32)
        
        return {"audio": [dummy_audio] * num_audios}


class FireRedASRMultiModalProcessor(
        EncDecMultiModalProcessor[FireRedASRProcessingInfo]):
    """Multi-modal processor for FireRedASR."""
    
    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=16000)  # 16kHz audio
        
    @property
    def pad_dummy_encoder_prompt(self) -> bool:
        return True
        
    def create_encoder_prompt(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
    ) -> Union[str, list[int]]:
        # Create dummy encoder prompt for audio features
        return [0]
        
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_data:
            # Process audio data
            audio_data = mm_data.get("audio")
            if audio_data is not None:
                # Convert audio to features (mel-spectrogram or similar)
                # This is a simplified version - in practice, you'd use proper audio processing
                if isinstance(audio_data, (list, tuple)):
                    audio_data = audio_data[0]
                if isinstance(audio_data, np.ndarray):
                    # Assume audio_data is already processed features
                    input_features = torch.from_numpy(audio_data).float()
                else:
                    # Create dummy features
                    input_features = torch.randn(1, 1000, 80)
                    
                return BatchFeature({"input_features": input_features})
                
        return BatchFeature({})
        
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(input_features=MultiModalFieldConfig.batched("audio"))
        
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        num_tokens = self.info.get_num_audio_tokens()
        return [PromptReplacement(
            modality="audio",
            target="<|audio|>",
            replacement=[0] * num_tokens,  # Dummy tokens
        )]


@MULTIMODAL_REGISTRY.register_processor(FireRedASRMultiModalProcessor,
                                        info=FireRedASRProcessingInfo,
                                        dummy_inputs=FireRedASRDummyInputsBuilder)
class FireRedASRForConditionalGeneration(nn.Module, SupportsTranscription,
                                        SupportsMultiModal):
    """FireRedASR model for conditional generation with transcription support."""
    
    packed_modules_mapping = {
        "self_attn.qkv_proj": [
            "self_attn.q_proj",
            "self_attn.k_proj", 
            "self_attn.v_proj",
        ],
    }
    
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_substr={
        ".linear1.": ".feed_forward.linear1.",
        ".linear2.": ".feed_forward.linear2.",
    })
    
    # FireRedASR supports transcription only
    supports_transcription_only = True
    supported_languages = FIREREDASR_SUPPORTED_LANGS
    
    @classmethod
    def validate_language(cls, language: Optional[str]) -> Optional[str]:
        if language is None:
            logger.warning(
                "Defaulting to language='zh'. If you wish to transcribe audio "
                "in a different language, pass the `language` field."
            )
            language = "zh"
        return super().validate_language(language)
        
    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
        language: Optional[str],
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: Optional[str],
    ) -> PromptType:
        if language is None:
            raise ValueError("Language must be specified for FireRedASR")
            
        # FireRedASR uses encoder-decoder architecture similar to Whisper
        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "audio": (audio, stt_config.sample_rate),
                },
            },
            "decoder_prompt": (
                (f"<|prev|>{request_prompt}" if request_prompt else "")
                + f"<|startoftranscript|><|{language}|>"
                + f"<|{task_type}|><|notimestamps|>"
            ),
        }
        return cast(PromptType, prompt)
        
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("audio"):
            return None
        raise ValueError("Only audio modality is supported")
        
    @classmethod
    def get_speech_to_text_config(
        cls,
        model_config: ModelConfig,
        task_type: str,
    ) -> SpeechToTextConfig:
        return SpeechToTextConfig(
            sample_rate=16000,  # FireRedASR uses 16kHz
            max_audio_clip_s=60,  # FireRedASR supports up to 60s
            overlap_chunk_second=1,
            min_energy_split_window_size=1600,  # 100ms at 16kHz
        )
        
    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> Optional[int]:
        # Estimate tokens based on audio duration
        # FireRedASR typically processes ~50 frames per second
        return int(audio_duration_s * 50)
        
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        
        self.model = FireRedASRModel(vllm_config=vllm_config, prefix=prefix)
        
        # Output projection
        self.proj_out = ParallelLMHead(
            config.vocab_size,
            config.d_model,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.proj_out"
        )
        
        # Logits processor
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            config.vocab_size,
            scale=1.0
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        
        # Get encoder outputs
        encoder_outputs = self.model.get_encoder_outputs(
            audio_input.get("input_features"),
            audio_input.get("input_lengths")
        )
        
        if encoder_outputs is None:
            raise ValueError("Audio input is required for FireRedASR")
            
        encoder_output, _, encoder_mask = encoder_outputs
        
        # Decoder forward pass
        hidden_states = self.model.decoder(input_ids, encoder_output, encoder_mask)
        
        return hidden_states
        
    def get_language_model(self) -> torch.nn.Module:
        return self.model.decoder
        
    def get_multimodal_embeddings(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        encoder_outputs = self.model.get_encoder_outputs(
            audio_input.get("input_features"),
            audio_input.get("input_lengths")
        )
        if encoder_outputs is None:
            return []
        return [encoder_outputs[0]]  # Return encoder output
        
    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        return self.model.decoder.embed_tokens(input_ids)
        
    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> FireRedASRAudioInputs:
        input_features = kwargs.pop("input_features", None)
        input_lengths = kwargs.pop("input_lengths", None)
        
        return FireRedASRAudioInputs(
            input_features=input_features,
            input_lengths=input_lengths,
        )
        
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = self.logits_processor(self.proj_out, hidden_states,
                                     sampling_metadata)
        return logits
        
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["proj_out."])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)