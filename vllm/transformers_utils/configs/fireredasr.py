# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FireRedASR model configuration."""

from transformers import PretrainedConfig


class FireRedASRConfig(PretrainedConfig):
    """
    Configuration class for FireRedASR model.

    FireRedASR is an Attention-based Encoder-Decoder (AED) model for
    Automatic Speech Recognition, specifically designed for Chinese and
    English speech recognition.

    Args:
        vocab_size (`int`, *optional*, defaults to 8000):
            Vocabulary size of the model. Defines the number of different
            tokens that can be represented.
        d_model (`int`, *optional*, defaults to 512):
            Dimensionality of the encoder and decoder layers.
        n_layers_enc (`int`, *optional*, defaults to 12):
            Number of Conformer encoder layers.
        n_layers_dec (`int`, *optional*, defaults to 6):
            Number of Transformer decoder layers.
        n_head (`int`, *optional*, defaults to 8):
            Number of attention heads in the multi-head attention layers.
        kernel_size (`int`, *optional*, defaults to 31):
            Kernel size for the convolution module in Conformer blocks.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers.
        residual_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for residual connections.
        pe_maxlen (`int`, *optional*, defaults to 5000):
            Maximum length for positional encoding.
        idim (`int`, *optional*, defaults to 80):
            Input feature dimension (e.g., number of mel-filterbank features).
        odim (`int`, *optional*, defaults to 8000):
            Output dimension (vocabulary size).
        sos_id (`int`, *optional*, defaults to 1):
            Start-of-sequence token ID.
        eos_id (`int`, *optional*, defaults to 2):
            End-of-sequence token ID.
        pad_id (`int`, *optional*, defaults to 0):
            Padding token ID.

    Example:
        ```python
        >>> from vllm.transformers_utils.configs.fireredasr import FireRedASRConfig
        >>>
        >>> # Initializing a FireRedASR configuration
        >>> configuration = FireRedASRConfig()
        >>>
        >>> # Initializing a model from the configuration
        >>> from vllm import LLM
        >>> model = LLM(model="path/to/fireredasr-model")
        ```
    """

    model_type = "fireredasr_aed"

    def __init__(
        self,
        vocab_size: int = 8000,
        d_model: int = 512,
        n_layers_enc: int = 12,
        n_layers_dec: int = 6,
        n_head: int = 8,
        kernel_size: int = 31,
        dropout_rate: float = 0.1,
        residual_dropout: float = 0.1,
        pe_maxlen: int = 5000,
        idim: int = 80,
        odim: int = 8000,
        sos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers_enc = n_layers_enc
        self.n_layers_dec = n_layers_dec
        self.n_head = n_head
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.residual_dropout = residual_dropout
        self.pe_maxlen = pe_maxlen
        self.idim = idim
        self.odim = odim
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
