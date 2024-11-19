import torch
from torch import Tensor
from typing import Callable
from loguru import logger


def _patch_sdpa(
    patch_func: Callable[[Tensor, Tensor, Tensor, Tensor | None, float, bool, float | None], Tensor],
):
    """(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)"""

    torch_sdpa = torch.nn.functional.scaled_dot_product_attention

    def sdpa_hijack_flash(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        try:
            return patch_func(query, key, value, attn_mask, dropout_p, is_causal, scale)
        except Exception:
            hidden_states = torch_sdpa(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )
        return hidden_states

    torch.nn.functional.scaled_dot_product_attention = sdpa_hijack_flash


try:
    from flash_attn import flash_attn_func

    def sdpa_hijack_flash(q, k, v, m, p, c, s):
        assert m is None
        result = flash_attn_func(
            q=q.transpose(1, 2),
            k=k.transpose(1, 2),
            v=v.transpose(1, 2),
            dropout_p=p,
            softmax_scale=s if s else q.shape[-1] ** (-0.5),
            causal=c,
        )
        assert isinstance(result, Tensor)
        return result.transpose(1, 2)

    _patch_sdpa(sdpa_hijack_flash)
    logger.debug("# # # Patched SDPA with Flash Attention # # #")
except ImportError as e:
    logger.debug(f"# # # Could not load Flash Attention for hijack: {e} # # #")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
