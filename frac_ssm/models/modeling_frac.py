"""Hugging Face model implementation for Frac SSM."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from frac_ssm.layers import frac_compute_scan_params, frac_skip_mix
from frac_ssm.models.configuration_frac import FracConfig
from frac_ssm.modules.rmsnorm import RMSNorm
from frac_ssm.modules.rmsnorm_gated import MambaRMSNormGated as RMSNormGated
from frac_ssm.ops.frac_scan import frac_scan


logger = logging.get_logger(__name__)

LayerCache = tuple[torch.Tensor, torch.Tensor]
PastKeyValues = tuple[LayerCache | None, ...]


def _mark_no_weight_decay(parameter: nn.Parameter) -> None:
    parameter._no_weight_decay = True


class FracPreTrainedModel(PreTrainedModel):
    config_class = FracConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FracBlock"]

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (RMSNorm, RMSNormGated)):
            nn.init.ones_(module.weight)

class FracMixer(nn.Module):
    def __init__(self, config: FracConfig):
        super().__init__()

        if config.hidden_act not in ("silu", "swish"):
            raise ValueError("FracMixer only supports silu/swish activation.")

        self.config = config
        self.hidden_size = config.hidden_size
        self.inner_dim = config.intermediate_size
        self.num_heads = config.num_attention_heads
        self.num_modes = config.frac_num_modes
        self.head_dim = config.frac_head_dim
        self.conv_kernel = config.frac_conv_kernel
        self.chunk_size = config.frac_chunk_size
        self.read_temp = 1.0
        self.act = ACT2FN[config.hidden_act]

        self.in_proj = nn.Linear(self.hidden_size, 2 * self.inner_dim, bias=False)
        self.conv1d = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=self.conv_kernel,
            groups=self.inner_dim,
            bias=config.frac_conv_bias,
            padding=self.conv_kernel - 1,
        )

        self.dyn_proj = nn.Linear(self.inner_dim, 3 * self.num_heads, bias=True)
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads))
        _mark_no_weight_decay(self.dt_bias)

        self.rw_proj = nn.Linear(self.inner_dim, 2 * self.num_heads * self.num_modes, bias=False)

        self.read_residual_logit = nn.Parameter(torch.tensor([float(config.frac_residual_read_init)]))
        _mark_no_weight_decay(self.read_residual_logit)

        write_scale_init = float(config.frac_write_scale_init)
        write_gate_prob = min(max(write_scale_init, 1e-4), 1.0 - 1e-4)
        write_gate_logit = math.log(write_gate_prob / (1.0 - write_gate_prob))
        self.write_residual_scale = nn.Parameter(torch.tensor([write_gate_logit], dtype=torch.float32))
        _mark_no_weight_decay(self.write_residual_scale)

        base_log_tau = torch.linspace(math.log(config.frac_tau_min), math.log(config.frac_tau_max), self.num_modes, dtype=torch.float32)
        self.register_buffer("base_log_tau", base_log_tau, persistent=True)

        self.out_norm = RMSNormGated(self.inner_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.inner_dim, self.hidden_size, bias=False)

        self.D = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        _mark_no_weight_decay(self.D)

    def _causal_conv_full(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x_transposed = x.transpose(1, 2)
        if attention_mask is not None:
            x_transposed = x_transposed * attention_mask[:, None].to(x_transposed.dtype)

        output = self.conv1d(x_transposed)[..., : x.shape[1]]
        output = self.act(output).transpose(1, 2)
        output = output.contiguous()

        if self.conv_kernel == 1:
            conv_state = x_transposed.new_zeros(x_transposed.shape[0], x_transposed.shape[1], 0)
        else:
            state_length = self.conv_kernel - 1
            left_padding = max(state_length - x_transposed.shape[-1], 0)
            conv_state = F.pad(x_transposed, (left_padding, 0))[..., -state_length:].contiguous()

        return output, conv_state

    def _causal_conv_step(self, x: torch.Tensor, conv_state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.conv_kernel == 1:
            output = x * self.conv1d.weight[:, 0, 0][None]
            if self.conv1d.bias is not None:
                output = output + self.conv1d.bias[None]
            return self.act(output), x.new_zeros(x.shape[0], x.shape[1], 0)

        if conv_state is None:
            conv_state = x.new_zeros(x.shape[0], x.shape[1], self.conv_kernel - 1)

        window = torch.cat((conv_state, x.unsqueeze(-1)), dim=-1)
        output = (window * self.conv1d.weight[:, 0][None]).sum(dim=-1)
        if self.conv1d.bias is not None:
            output = output + self.conv1d.bias[None]

        return self.act(output), window[:, :, 1:].contiguous()

    def _update_state(
        self,
        u: torch.Tensor,
        rho: torch.Tensor,
        kappa: torch.Tensor,
        c: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update one token with state `[B, H, M, R]`.

        ``u`` has shape ``[B, H, R]``; ``rho``, ``kappa``, and ``c`` have
        shape ``[B, H, M]``.
        """
        if state is None:
            state = u.new_zeros(u.shape[0], self.num_heads, self.num_modes, self.head_dim, dtype=torch.float32)
        else:
            state = state.float()

        state = rho.unsqueeze(-1) * state + u.float().unsqueeze(2) * kappa.unsqueeze(-1)
        mixed = (c.unsqueeze(-1) * state).sum(dim=2)
        return mixed, state

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layer_past: LayerCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, LayerCache | None]:
        batch_size, sequence_length, _ = hidden_states.shape
        past_state = layer_past[0] if layer_past is not None else None
        past_conv_state = layer_past[1] if layer_past is not None else None

        x, gate = self.in_proj(hidden_states).chunk(2, dim=-1)

        if self.training or (layer_past is None and sequence_length > 1):
            x_conv, conv_state = self._causal_conv_full(x, attention_mask)

            # Cache state is [B, H, M, R]; the scan boundary is [B * H, R, M].
            initial_state = None
            if past_state is not None:
                initial_state = past_state.permute(0, 1, 3, 2).reshape(batch_size * self.num_heads, self.head_dim, self.num_modes).contiguous()

            u, c, kappa, rho = frac_compute_scan_params(self, x_conv, x, attention_mask)
            c = c.to(u.dtype)
            kappa = kappa.to(u.dtype)
            scan_output = frac_scan(u.contiguous(), c.contiguous(), kappa.contiguous(), rho.contiguous(), self.chunk_size, initial_state, use_cache)

            next_cache = None
            if use_cache:
                mixed, final_state = scan_output
                final_state = final_state.view(batch_size, self.num_heads, self.head_dim, self.num_modes).permute(0, 1, 3, 2).contiguous()
                next_cache = (final_state, conv_state)
            else:
                mixed = scan_output

            mixed = frac_skip_mix(self, mixed, x_conv)
            mixed = self.out_norm(mixed, gate=gate + self.config.frac_gate_bias_init)
            return self.out_proj(mixed), next_cache

        outputs = []
        state = past_state
        conv_state = past_conv_state

        for token_idx in range(sequence_length):
            x_token = x[:, token_idx]
            step_mask = attention_mask[:, token_idx:token_idx + 1] if attention_mask is not None else None
            conv_input = x_token
            if step_mask is not None:
                conv_input = conv_input * step_mask.to(conv_input.dtype)

            x_conv, conv_state = self._causal_conv_step(conv_input, conv_state)
            u, c, kappa, rho = frac_compute_scan_params(self, x_conv[:, None], x_token[:, None], step_mask)

            u = u.view(batch_size, self.num_heads, self.head_dim)
            c = c.view(batch_size, self.num_heads, self.num_modes)
            kappa = kappa.view(batch_size, self.num_heads, self.num_modes)
            rho = rho.view(batch_size, self.num_heads, self.num_modes)

            mixed, state = self._update_state(u, rho, kappa, c, state)
            mixed = mixed + u.to(mixed.dtype) * self.D[None].to(mixed.dtype)
            mixed = mixed.reshape(batch_size, self.inner_dim).to(hidden_states.dtype)
            mixed = self.out_norm(mixed, gate=gate[:, token_idx] + self.config.frac_gate_bias_init)
            outputs.append(self.out_proj(mixed))

        output = torch.stack(outputs, dim=1)
        next_cache = (state, conv_state) if use_cache else None
        return output, next_cache


class FracBlock(nn.Module):
    def __init__(self, config: FracConfig):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mixer = FracMixer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layer_past: LayerCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, LayerCache | None]:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states, next_cache = self.mixer(hidden_states, attention_mask=attention_mask, layer_past=layer_past, use_cache=use_cache)
        return residual + hidden_states, next_cache


class FracModel(FracPreTrainedModel):
    def __init__(self, config: FracConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing = False

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList(FracBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast:
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        return_dict = self.config.return_dict if return_dict is None else return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds.")

        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if attention_mask is None:
            attention_mask = hidden_states.new_ones(hidden_states.shape[:2], dtype=torch.long)

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        all_hidden_states = [] if output_hidden_states else None
        next_past = [] if use_cache else None

        if past_key_values is None:
            past_key_values = (None,) * len(self.layers)

        for layer_idx, layer in enumerate(self.layers):
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)

            layer_past = past_key_values[layer_idx]
            if self.gradient_checkpointing and self.training:

                # Bind this block for backward recomputation after the loop ends.
                def custom_forward(
                    hidden_states_in: torch.Tensor,
                    attention_mask_in: torch.Tensor,
                    layer: FracBlock = layer,
                ) -> torch.Tensor:
                    return layer(hidden_states_in, attention_mask=attention_mask_in, layer_past=None, use_cache=False)[0]

                hidden_states = self._gradient_checkpointing_func(custom_forward, hidden_states, attention_mask)
                layer_cache = None
            else:
                hidden_states, layer_cache = layer(hidden_states, attention_mask=attention_mask, layer_past=layer_past, use_cache=use_cache)

            if next_past is not None:
                next_past.append(layer_cache)

        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        if not return_dict:
            outputs = (hidden_states,)
            if next_past is not None:
                outputs += (tuple(next_past),)
            if all_hidden_states is not None:
                outputs += (tuple(all_hidden_states),)
            return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=tuple(next_past) if next_past is not None else None,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
        )


class FracForCausalLM(FracPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False

    def __init__(self, config: FracConfig):
        super().__init__(config)
        self.model = FracModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: nn.Module) -> None:
        self.model = decoder

    def get_decoder(self) -> nn.Module:
        return self.model

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: PastKeyValues | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> dict:
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -1:]
            inputs_embeds = None

        model_inputs = {"inputs_embeds": inputs_embeds} if inputs_embeds is not None else {"input_ids": input_ids}
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "use_cache": use_cache,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values: PastKeyValues | None, beam_idx: torch.LongTensor) -> PastKeyValues | None:
        if past_key_values is None:
            return None

        reordered = []
        for layer_past in past_key_values:
            if layer_past is None:
                reordered.append(None)
                continue
            state, conv_state = layer_past
            reordered.append((state.index_select(0, beam_idx), conv_state.index_select(0, beam_idx)))
        return tuple(reordered)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        return_dict = self.config.return_dict if return_dict is None else return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), ignore_index=-100)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )


__all__ = ["FracBlock", "FracForCausalLM", "FracMixer", "FracModel", "FracPreTrainedModel"]
