from transformers.configuration_utils import PretrainedConfig


class FracConfig(PretrainedConfig):
    model_type = "frac"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        intermediate_size: int | None = None,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 12,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        frac_expand: float = 2.0,
        frac_num_modes: int = 8,
        frac_conv_kernel: int = 4,
        frac_dt_min: float = 1e-4,
        frac_dt_max: float = 1.0,
        frac_alpha_min: float = 0.20,
        frac_alpha_max: float = 0.95,
        frac_tau_min: float = 1.0,
        frac_tau_max: float = 65536.0,
        frac_chunk_size: int = 32,
        frac_lambda_min: float = 0.25,
        frac_lambda_max: float = 4.0,
        frac_residual_read_init: float = -2.0,
        frac_write_scale_init: float = 0.25,
        frac_gate_bias_init: float = 0.0,
        frac_conv_bias: bool = True,
        frac_head_dim: int | None = None,
        return_dict: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = (
            int(hidden_size * frac_expand)
            if intermediate_size is None
            else intermediate_size
        )
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        self.frac_expand = frac_expand
        self.frac_num_modes = frac_num_modes
        self.frac_conv_kernel = frac_conv_kernel
        self.frac_dt_min = frac_dt_min
        self.frac_dt_max = frac_dt_max
        self.frac_alpha_min = frac_alpha_min
        self.frac_alpha_max = frac_alpha_max
        self.frac_tau_min = frac_tau_min
        self.frac_tau_max = frac_tau_max
        self.frac_chunk_size = frac_chunk_size
        self.frac_lambda_min = frac_lambda_min
        self.frac_lambda_max = frac_lambda_max
        self.frac_residual_read_init = frac_residual_read_init
        self.frac_write_scale_init = frac_write_scale_init
        self.frac_gate_bias_init = frac_gate_bias_init
        self.frac_conv_bias = frac_conv_bias

        if not isinstance(frac_chunk_size, int) or isinstance(frac_chunk_size, bool) or frac_chunk_size <= 0:
            raise ValueError(
                "frac_chunk_size must be a positive integer, "
                f"got {frac_chunk_size}."
            )

        inferred_inner = self.intermediate_size
        if frac_head_dim is None:
            if inferred_inner % num_attention_heads != 0:
                raise ValueError(
                    f"intermediate_size={inferred_inner} must be divisible by "
                    f"num_attention_heads={num_attention_heads}."
                )
            frac_head_dim = inferred_inner // num_attention_heads
        else:
            if frac_head_dim * num_attention_heads != inferred_inner:
                raise ValueError(
                    "frac_head_dim * num_attention_heads must equal intermediate_size. "
                    f"Got {frac_head_dim} * {num_attention_heads} != {inferred_inner}."
                )
        self.frac_head_dim = frac_head_dim

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            return_dict=return_dict,
            **kwargs,
        )


__all__ = ["FracConfig"]
