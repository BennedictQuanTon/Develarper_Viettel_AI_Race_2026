"""Payload p09_shortconv_v2 — ShortConv decode fuse (general state_len>=2)."""

from __future__ import annotations

import torch

from develarper_opt.kernels.shortconv_fuse import fused_shortconv_decode
from develarper_opt.platform.registry import PayloadResult, register


def _make_forward_cuda(original_forward_cuda):
    def forward_cuda(self, hidden_states: torch.Tensor, output: torch.Tensor):
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_fn,
            causal_conv1d_update,
        )
        from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionMetadata

        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        attn_metadata = None
        if attn_metadata_raw is not None:
            assert isinstance(attn_metadata_raw, dict)
            attn_metadata = attn_metadata_raw[self.prefix]
            assert isinstance(attn_metadata, ShortConvAttentionMetadata)

        conv_state = (
            self.kv_cache[0]
            if is_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )
        state_indices_tensor_p = (
            attn_metadata.state_indices_tensor_p if attn_metadata is not None else None
        )
        state_indices_tensor_d = (
            attn_metadata.state_indices_tensor_d if attn_metadata is not None else None
        )
        has_initial_states_p = (
            attn_metadata.has_initial_states_p if attn_metadata is not None else None
        )
        query_start_loc_p = (
            attn_metadata.query_start_loc_p if attn_metadata is not None else None
        )

        BCx, _ = self.in_proj(hidden_states)
        B, C, x = BCx.chunk(3, dim=-1)
        conv_weights = self.conv.weight.view(
            self.conv.weight.size(0), self.conv.weight.size(2)
        )

        if attn_metadata is None:
            Bx = (B * x).contiguous()
            hidden_states_p = C * Bx
            contextualized_states, _ = self.out_proj(hidden_states_p)
            return contextualized_states

        num_prefills = attn_metadata.num_prefills
        num_decodes = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        num_actual_tokens = num_decodes + num_prefill_tokens

        B_d, B_p = torch.split(
            B[:num_actual_tokens], [num_decodes, num_prefill_tokens], dim=0
        )
        C_d, C_p = torch.split(
            C[:num_actual_tokens], [num_decodes, num_prefill_tokens], dim=0
        )
        x_d, x_p = torch.split(
            x[:num_actual_tokens], [num_decodes, num_prefill_tokens], dim=0
        )
        conv_output_list = []

        if has_prefill:
            Bx_p = (B_p * x_p).transpose(0, 1)
            Bx = causal_conv1d_fn(
                Bx_p,
                conv_weights,
                self.conv.bias,
                activation=None,
                conv_states=conv_state,
                has_initial_state=has_initial_states_p,
                cache_indices=state_indices_tensor_p,
                metadata=attn_metadata,
                query_start_loc=query_start_loc_p,
            ).transpose(0, 1)[:num_prefill_tokens]
            conv_output_list.append(C_p * Bx)

        if has_decode:
            used_fuse = False
            if (
                state_indices_tensor_d is not None
                and conv_weights.shape[-1] == 3
                and conv_state.dim() == 3
                and conv_state.shape[-1] >= 2
                and B_d.is_cuda
            ):
                try:
                    y = fused_shortconv_decode(
                        B_d.contiguous(),
                        x_d.contiguous(),
                        C_d.contiguous(),
                        conv_state,
                        conv_weights.contiguous(),
                        self.conv.bias,
                        state_indices_tensor_d.flatten(),
                    )
                    conv_output_list.insert(0, y)
                    used_fuse = True
                except Exception:
                    used_fuse = False
            if not used_fuse:
                Bx_d = (B_d * x_d).contiguous()
                Bx = causal_conv1d_update(
                    Bx_d,
                    conv_state,
                    conv_weights,
                    self.conv.bias,
                    activation=None,
                    conv_state_indices=state_indices_tensor_d,
                )
                conv_output_list.insert(0, C_d * Bx)

        hidden_states_out = torch.vstack(conv_output_list)
        output[:num_actual_tokens], _ = self.out_proj(hidden_states_out)

    forward_cuda.__develarper_p09_sc__ = True  # type: ignore[attr-defined]
    forward_cuda.__wrapped_original__ = original_forward_cuda  # type: ignore[attr-defined]
    return forward_cuda


@register("p09_shortconv_v2")
def apply_p09_shortconv_v2() -> PayloadResult:
    try:
        from vllm.model_executor.layers.mamba.short_conv import ShortConv
    except Exception as exc:
        return PayloadResult(
            name="p09_shortconv_v2",
            status="skipped",
            reason=f"ShortConv import failed: {exc!r}",
        )

    if getattr(ShortConv.forward_cuda, "__develarper_p09_sc__", False):
        return PayloadResult(
            name="p09_shortconv_v2",
            status="already_applied",
            reason="forward_cuda already patched",
        )

    if not callable(getattr(ShortConv, "forward_cuda", None)):
        return PayloadResult(
            name="p09_shortconv_v2",
            status="skipped",
            reason="ShortConv.forward_cuda missing",
        )

    try:
        from vllm.model_executor.layers.mamba.ops import causal_conv1d as _cc

        if not hasattr(_cc, "causal_conv1d_update") or not hasattr(_cc, "causal_conv1d_fn"):
            return PayloadResult(
                name="p09_shortconv_v2",
                status="skipped",
                reason="causal_conv1d ops missing",
            )
    except Exception as exc:
        return PayloadResult(
            name="p09_shortconv_v2",
            status="skipped",
            reason=f"causal_conv1d import failed: {exc!r}",
        )

    original = ShortConv.forward_cuda
    ShortConv.forward_cuda = _make_forward_cuda(original)  # type: ignore[method-assign]
    return PayloadResult(
        name="p09_shortconv_v2",
        status="applied",
        reason="ShortConv.forward_cuda patched v2 (state_len>=2)",
        details={"width_target": "3", "state_len": ">=2"},
    )
