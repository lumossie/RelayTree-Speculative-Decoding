"""Load the Tree-Attention backends used in the RelayTree paper."""

from __future__ import annotations

from typing import Type

from transformers import AutoConfig


def get_tree_attn_causallm_class(
    model_name_or_path: str,
    *,
    local_files_only: bool = False,
) -> Type:
    """Return the patched causal-LM class for Pythia or Qwen2 checkpoints."""
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )
    model_type = getattr(config, "model_type", None)

    if model_type == "gpt_neox":
        from relaytree.models.gpt_neox.modeling import GPTNeoXForCausalLM

        return GPTNeoXForCausalLM

    if model_type == "qwen2":
        from relaytree.models.qwen2.modeling import Qwen2ForCausalLM

        return Qwen2ForCausalLM

    raise ValueError(
        "The publication release contains Tree-Attention backends only for "
        f"Pythia/GPT-NeoX and Qwen2 (received model_type={model_type!r})."
    )
