"""Tree-Attention patched GPT-NeoX/Pythia model.

Tree-attention patched GPT-NeoX model.

This module provides a drop-in replacement for `transformers` GPT-NeoX
models that supports the extra keyword arguments used by Tree-MCSD:
  - tree_attn_mask: a boolean (tgt_len, src_len) mask to override the causal mask
  - position_ids: custom position indices for tree decoding

The implementation is intentionally minimal: it reuses the upstream
`transformers.models.gpt_neox` components and only patches the causal
mask path. GPT-NeoX already supports explicit `position_ids`; we only
normalize 1D inputs to the batched format expected by the model.
"""

from .modeling import GPTNeoXForCausalLM, GPTNeoXModel

__all__ = ["GPTNeoXForCausalLM", "GPTNeoXModel"]
