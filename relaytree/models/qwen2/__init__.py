"""Tree-Attention patched Qwen2 model.

Tree-attention patched Qwen2 model.

This module wraps the Hugging Face Qwen2 implementation and adds
Tree-MCSD's extra `tree_attn_mask` pathway while preserving explicit
`position_ids` handling.
"""

from .modeling import Qwen2ForCausalLM, Qwen2Model

__all__ = ["Qwen2ForCausalLM", "Qwen2Model"]
