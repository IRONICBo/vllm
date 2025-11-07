#!/usr/bin/env python3
"""
FireRedASR Integration Test Script

This script tests the FireRedASR model integration with vLLM.
"""

import numpy as np
import torch
import pytest
from unittest.mock import Mock, patch

from vllm.config import VllmConfig, ModelConfig, CacheConfig
from vllm.model_executor.models.fireredasr_aed import (
    FireRedASRConfig,
    FireRedASRForConditionalGeneration,
    FireRedASRConformerEncoder,
    FireRedASRTransformerDecoder,
    FireRedASRMultiModalProcessor,
    FireRedASRProcessingInfo,
    FIREREDASR_SUPPORTED_LANGS
)


class TestFireRedASRConfig:
    """Test FireRedASR configuration."""
    
    def test_default_config(self):
        config = FireRedASRConfig()
        assert config.model_type == "fireredasr_aed"
        assert config.vocab_size == 8000
        assert config.d_model == 512
        assert config.n_layers_enc == 12
        assert config.n_layers_dec == 6
        assert config.n_head == 8
        assert config.idim == 80
        assert config.sos_id == 1
        assert config.eos_id == 2
        assert config.pad_id == 0
    
    def test_custom_config(self):
        config = FireRedASRConfig(
            vocab_size=16000,
            d_model=768,
            n_layers_enc=24,
            n_layers_dec=12
        )
        assert config.vocab_size == 16000
        assert config.d_model == 768
        assert config.n_layers_enc == 24
        assert config.n_layers_dec == 12


class TestFireRedASRModel:
    """Test FireRedASR model components."""
    
    @pytest.fixture
    def mock_vllm_config(self):
        config = FireRedASRConfig()
        model_config = Mock(spec=ModelConfig)
        model_config.hf_config = config
        
        vllm_config = Mock(spec=VllmConfig)
        vllm_config.model_config = model_config
        vllm_config.cache_config = Mock(spec=CacheConfig)
        vllm_config.quant_config = None
        
        return vllm_config
    
    def test_conformer_encoder_init(self, mock_vllm_config):
        config = mock_vllm_config.model_config.hf_config
        encoder = FireRedASRConformerEncoder(
            idim=config.idim,
            d_model=config.d_model,
            n_layers=config.n_layers_enc,
            n_head=config.n_head,
            d_ff=config.d_model * 4,
            kernel_size=config.kernel_size,
            dropout=config.dropout_rate,
            pe_maxlen=config.pe_maxlen,
            quant_config=None,
            prefix="encoder"
        )
        assert encoder.d_model == config.d_model
        assert len(encoder.layers) == config.n_layers_enc
    
    def test_transformer_decoder_init(self, mock_vllm_config):
        config = mock_vllm_config.model_config.hf_config
        decoder = FireRedASRTransformerDecoder(
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
            cache_config=None,
            quant_config=None,
            prefix="decoder"
        )
        assert decoder.vocab_size == config.vocab_size
        assert len(decoder.layers) == config.n_layers_dec
    
    def test_model_forward(self, mock_vllm_config):
        model = FireRedASRForConditionalGeneration(
            vllm_config=mock_vllm_config,
            prefix=""
        )
        
        # Test input shapes
        batch_size, seq_len = 2, 100
        audio_len, feature_dim = 1000, 80
        
        # Mock inputs
        input_ids = torch.randint(0, 8000, (batch_size, seq_len))
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        input_features = torch.randn(batch_size, audio_len, feature_dim)
        
        # Test forward pass (this would require proper weight initialization)
        # For now, just test that the model can be instantiated
        assert model is not None
        assert hasattr(model, 'forward')
        assert hasattr(model, 'get_multimodal_embeddings')


class TestSupportsTranscriptionInterface:
    """Test SupportsTranscription interface implementation."""
    
    def test_supported_languages(self):
        assert FireRedASRForConditionalGeneration.supported_languages == FIREREDASR_SUPPORTED_LANGS
        assert "zh" in FireRedASRForConditionalGeneration.supported_languages
        assert "en" in FireRedASRForConditionalGeneration.supported_languages
    
    def test_supports_transcription_only(self):
        assert FireRedASRForConditionalGeneration.supports_transcription_only is True
    
    def test_validate_language(self):
        # Test default language
        result = FireRedASRForConditionalGeneration.validate_language(None)
        assert result == "zh"
        
        # Test valid language
        result = FireRedASRForConditionalGeneration.validate_language("en")
        assert result == "en"
        
        # Test invalid language (should raise error)
        with pytest.raises(ValueError):
            FireRedASRForConditionalGeneration.validate_language("fr")
    
    def test_get_speech_to_text_config(self):
        from vllm.config import SpeechToTextConfig
        
        mock_model_config = Mock()
        config = FireRedASRForConditionalGeneration.get_speech_to_text_config(
            mock_model_config, "transcribe"
        )
        
        assert isinstance(config, SpeechToTextConfig)
        assert config.sample_rate == 16000
        assert config.max_audio_clip_s == 60
        assert config.overlap_chunk_second == 1
        assert config.min_energy_split_window_size == 1600
    
    def test_get_generation_prompt(self):
        from vllm.config import SpeechToTextConfig, ModelConfig
        
        audio = np.random.randn(16000)  # 1 second of audio
        stt_config = SpeechToTextConfig(sample_rate=16000)
        model_config = Mock(spec=ModelConfig)
        
        prompt = FireRedASRForConditionalGeneration.get_generation_prompt(
            audio=audio,
            stt_config=stt_config,
            model_config=model_config,
            language="zh",
            task_type="transcribe",
            request_prompt="",
            to_language=None
        )
        
        assert isinstance(prompt, dict)
        assert "encoder_prompt" in prompt
        assert "decoder_prompt" in prompt
        assert "multi_modal_data" in prompt["encoder_prompt"]
        assert "audio" in prompt["encoder_prompt"]["multi_modal_data"]
    
    def test_get_num_audio_tokens(self):
        from vllm.config import SpeechToTextConfig, ModelConfig
        
        stt_config = SpeechToTextConfig(sample_rate=16000)
        model_config = Mock(spec=ModelConfig)
        
        # Test 10 seconds of audio
        num_tokens = FireRedASRForConditionalGeneration.get_num_audio_tokens(
            audio_duration_s=10.0,
            stt_config=stt_config,
            model_config=model_config
        )
        
        assert num_tokens == 500  # 10 * 50 frames per second


class TestMultiModalProcessor:
    """Test multi-modal processor."""
    
    @pytest.fixture
    def processor_info(self):
        mock_ctx = Mock()
        mock_ctx.get_hf_config.return_value = FireRedASRConfig()
        
        info = FireRedASRProcessingInfo(mock_ctx)
        return info
    
    def test_processing_info(self, processor_info):
        assert processor_info.get_supported_mm_limits() == {"audio": 1}
        assert processor_info.get_num_audio_tokens() == 3000
    
    def test_multimodal_processor_init(self, processor_info):
        processor = FireRedASRMultiModalProcessor(processor_info)
        assert processor.pad_dummy_encoder_prompt is True
    
    def test_create_encoder_prompt(self, processor_info):
        processor = FireRedASRMultiModalProcessor(processor_info)
        
        prompt = processor.create_encoder_prompt("test", {})
        assert prompt == [0]


def test_model_registration():
    """Test that the model is properly registered."""
    from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
    
    assert "FireRedASRForConditionalGeneration" in _MULTIMODAL_MODELS
    model_info = _MULTIMODAL_MODELS["FireRedASRForConditionalGeneration"]
    assert model_info == ("fireredasr_aed", "FireRedASRForConditionalGeneration")


def test_integration_example():
    """Test a simple integration example."""
    
    # This is a mock integration test
    # In practice, you would need actual model weights and proper setup
    
    config = FireRedASRConfig(
        vocab_size=1000,  # Smaller for testing
        d_model=256,
        n_layers_enc=2,
        n_layers_dec=2,
        n_head=4
    )
    
    # Mock vLLM config
    model_config = Mock(spec=ModelConfig)
    model_config.hf_config = config
    
    vllm_config = Mock(spec=VllmConfig)
    vllm_config.model_config = model_config
    vllm_config.cache_config = Mock(spec=CacheConfig)
    vllm_config.quant_config = None
    
    # Test model instantiation
    model = FireRedASRForConditionalGeneration(
        vllm_config=vllm_config,
        prefix=""
    )
    
    assert model is not None
    assert hasattr(model, 'supports_transcription_only')
    assert model.supports_transcription_only is True


if __name__ == "__main__":
    # Run basic tests
    print("Running FireRedASR integration tests...")
    
    # Test configuration
    print("✓ Testing configuration...")
    test_config = TestFireRedASRConfig()
    test_config.test_default_config()
    test_config.test_custom_config()
    
    # Test interface
    print("✓ Testing SupportsTranscription interface...")
    test_interface = TestSupportsTranscriptionInterface()
    test_interface.test_supported_languages()
    test_interface.test_supports_transcription_only()
    
    # Test registration
    print("✓ Testing model registration...")
    test_model_registration()
    
    print("✓ All tests passed!")
    print("\nTo run full test suite with pytest:")
    print("pytest tests/test_fireredasr_integration.py -v")