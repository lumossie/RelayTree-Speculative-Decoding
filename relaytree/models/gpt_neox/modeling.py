# coding=utf-8
# Copyright 2026
#
# This file provides a minimal, tree-attention-compatible GPT-NeoX
# implementation. It reuses upstream `transformers.models.gpt_neox`
# components and only patches:
#   (1) causal-mask construction to accept a boolean `tree_attn_mask`, and
#   (2) compatibility shims so Tree-MCSD can access `.model` and `.lm_head`.

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXForCausalLM as _HFGPTNeoXForCausalLM,
    GPTNeoXModel as _HFGPTNeoXModel,
)


def _force_eager_tree_attn_config(config: GPTNeoXConfig) -> GPTNeoXConfig:
    config._attn_implementation = "eager"
    return config


def _tree_mask_to_additive_mask(
    tree_attn_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int,
    sequence_length: int,
    target_length: int,
) -> torch.Tensor:
    """Convert a bool allow-mask to the additive mask expected by eager attention."""
    if tree_attn_mask.dim() == 2:
        tree_attn_mask = tree_attn_mask.unsqueeze(0)
    if tree_attn_mask.dim() != 3:
        raise ValueError(f"tree_attn_mask should be 2D or 3D, got {tree_attn_mask.dim()}D")

    expected_shape = (sequence_length, target_length)
    if tuple(tree_attn_mask.shape[-2:]) != expected_shape:
        raise ValueError(
            f"tree_attn_mask should have shape {expected_shape}, got {tuple(tree_attn_mask.shape[-2:])}"
        )

    tree_attn_mask = tree_attn_mask.to(device=device, dtype=torch.bool)
    additive = torch.full(
        tree_attn_mask.shape,
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    additive.masked_fill_(tree_attn_mask, 0)

    if additive.size(0) == 1:
        additive = additive.expand(batch_size, -1, -1)
    elif additive.size(0) != batch_size:
        raise ValueError(
            f"tree_attn_mask batch dim mismatch: got {additive.size(0)} vs batch_size={batch_size}"
        )

    return additive.unsqueeze(1)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int) -> torch.Tensor:
    """Expand a 2D attention mask to 4D additive form, matching HF conventions."""
    bsz, src_len = mask.size()
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class GPTNeoXModel(_HFGPTNeoXModel):
    """Upstream GPTNeoXModel with a Tree-Attn causal-mask override."""

    def __init__(self, config: GPTNeoXConfig):
        _force_eager_tree_attn_config(config)
        super().__init__(config)
        self._tree_attn_mask: Optional[torch.Tensor] = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.FloatTensor]]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        tree_attn_mask: Optional[torch.Tensor] = None,
    ):
        if position_ids is not None:
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            position_ids = position_ids.long()

            if input_ids is not None:
                batch_size = input_ids.size(0)
            elif inputs_embeds is not None:
                batch_size = inputs_embeds.size(0)
            else:
                batch_size = None

            if batch_size is not None and position_ids.size(0) == 1 and batch_size > 1:
                position_ids = position_ids.expand(batch_size, -1)
            elif batch_size is not None and position_ids.size(0) != batch_size:
                raise ValueError(
                    f"position_ids batch dim mismatch: got {position_ids.size(0)} vs batch_size={batch_size}"
                )

        self._tree_attn_mask = tree_attn_mask
        try:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
        finally:
            self._tree_attn_mask = None

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        tree_attn_mask = self._tree_attn_mask
        if tree_attn_mask is None:
            return super()._update_causal_mask(
                attention_mask,
                input_tensor,
                cache_position,
                past_key_values,
                output_attentions,
            )

        dtype, device = input_tensor.dtype, input_tensor.device
        batch_size, sequence_length = input_tensor.shape[:2]
        if attention_mask is not None and attention_mask.dim() == 2:
            target_length = attention_mask.shape[-1]
        else:
            target_length = tree_attn_mask.shape[-1]

        causal_mask = _tree_mask_to_additive_mask(
            tree_attn_mask,
            dtype=dtype,
            device=device,
            batch_size=batch_size,
            sequence_length=sequence_length,
            target_length=target_length,
        )

        if attention_mask is None:
            return causal_mask

        if attention_mask.dim() == 2:
            expanded_mask = _expand_mask(
                attention_mask.to(device=device),
                dtype=dtype,
                tgt_len=sequence_length,
            )
            if expanded_mask.size(-1) != causal_mask.size(-1):
                raise ValueError(
                    f"attention_mask width mismatch: got {expanded_mask.size(-1)} vs "
                    f"tree_attn_mask width {causal_mask.size(-1)}"
                )
            return causal_mask + expanded_mask

        if attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                expanded_mask = torch.zeros_like(attention_mask, dtype=dtype, device=device)
                expanded_mask = expanded_mask.masked_fill(~attention_mask.to(device), torch.finfo(dtype).min)
            else:
                expanded_mask = attention_mask.to(device=device, dtype=dtype)

            if expanded_mask.size(0) == 1 and batch_size > 1:
                expanded_mask = expanded_mask.expand(batch_size, -1, -1, -1)
            if tuple(expanded_mask.shape[-2:]) != tuple(causal_mask.shape[-2:]):
                raise ValueError(
                    f"4D attention_mask shape mismatch: got {tuple(expanded_mask.shape[-2:])} vs "
                    f"{tuple(causal_mask.shape[-2:])}"
                )
            return causal_mask + expanded_mask

        raise ValueError(
            f"attention_mask should be 2D or 4D when tree_attn_mask is set, got {attention_mask.dim()}D"
        )


class GPTNeoXForCausalLM(_HFGPTNeoXForCausalLM):
    """GPTNeoXForCausalLM with Tree-Attn-compatible `.model` and `.lm_head`."""

    def __init__(self, config: GPTNeoXConfig):
        _force_eager_tree_attn_config(config)
        super().__init__(config)
        self.gpt_neox = GPTNeoXModel(config)
        self.post_init()

    @property
    def model(self):
        return self.gpt_neox

    @property
    def lm_head(self):
        return self.embed_out

    def get_input_embeddings(self):
        return self.gpt_neox.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.gpt_neox.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.FloatTensor]]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        tree_attn_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            tree_attn_mask=tree_attn_mask,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return CausalLMOutputWithPast(
            loss=None,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
