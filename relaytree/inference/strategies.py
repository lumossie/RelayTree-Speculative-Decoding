from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput

from relaytree.sampling.utils import lattice_based_quantization_torch


@dataclass
class DecoderOnlyDraftOutput(ModelOutput):
    """Decoder-only draft output."""

    sequences: torch.LongTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cand_probs: Optional[Tuple[torch.FloatTensor]] = None
    output_token_count: Optional[int] = None


@dataclass
class DecoderOnlyDraftReplayOutput(ModelOutput):
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    next_logits: Optional[torch.FloatTensor] = None


@dataclass
class DecoderOnlyVerificationOutput(ModelOutput):
    """Decoder-only verification output."""

    sequences: torch.LongTensor = None
    target_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    draft_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    acceptance_count: Optional[int] = None


def _MCSSwoReplacement(
    ground_probs: torch.FloatTensor,
    cand_probs: torch.FloatTensor,
    cand_tokens: torch.LongTensor,
) -> Optional[int]:
    cand_probs = cand_probs.to(device=ground_probs.device, dtype=torch.float32)
    ground_probs = ground_probs.to(dtype=torch.float32)
    if not torch.isfinite(cand_probs).all():
        cand_probs = _safe_normalize_probs(cand_probs)
    if not torch.isfinite(ground_probs).all():
        ground_probs.copy_(_safe_normalize_probs(ground_probs))

    p0 = ground_probs.clone()
    ground_probs.copy_(p0)

    for check_idx, cand_token in enumerate(cand_tokens):
        denom = cand_probs[cand_token]

        if (not torch.isfinite(denom)) or (denom <= 0):
            accept_threshold = torch.tensor(
                1.0 if ground_probs[cand_token] > 0 else 0.0,
                device=ground_probs.device,
            )
        else:
            accept_threshold = ground_probs[cand_token] / denom

        if torch.rand(1, device=accept_threshold.device) <= accept_threshold:
            return check_idx

        residual = torch.relu(ground_probs - cand_probs)
        s = residual.sum()

        if (not torch.isfinite(s)) or (s <= 0):
            ground_probs.copy_(p0)
            return None

        ground_probs.copy_(residual / s)

        cand_probs[cand_token] = 0
        s2 = cand_probs.sum()
        if (not torch.isfinite(s2)) or (s2 <= 0):
            V = cand_probs.numel()
            cand_probs[:] = 1.0 / float(V)
        else:
            cand_probs /= s2

    return None


def _safe_normalize_probs(probs: torch.Tensor) -> torch.Tensor:
    """Ensure probs is finite, non-negative, and sums to 1 on the last dim."""
    probs = probs.to(dtype=torch.float32)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = torch.clamp(probs, min=0.0)
    denom = probs.sum(dim=-1, keepdim=True)
    V = probs.size(-1)
    uniform = torch.full_like(probs, 1.0 / float(V))
    probs = torch.where(denom > 0, probs / denom, uniform)
    return probs


def _softmax_probs(logits: torch.Tensor, temp: float) -> torch.Tensor:
    probs = torch.softmax(logits.float() / temp, dim=-1)
    if torch.isfinite(probs).all():
        return probs
    return _safe_normalize_probs(probs)


class Strategy:
    """Shared static-quantization state for the paper's TreeStrategy."""

    def __init__(
        self,
        draft_model,
        target_model,
        k_config: Tuple[int, ...],
        draft_model_temp: float = 1.0,
        target_model_temp: float = 1.0,
        q_level: int = 0,
    ) -> None:
        self.k_config = tuple(k_config)
        self.draft_model = draft_model
        self.target_model = target_model
        self.draft_model_device = draft_model.model.get_input_embeddings().weight.device
        self.target_model_device = target_model.model.get_input_embeddings().weight.device
        self.max_draft_len = len(self.k_config)
        self.draft_model_temp = float(draft_model_temp)
        self.target_model_temp = float(target_model_temp)
        self.q_level = int(q_level)
        self.acceptance_check = _MCSSwoReplacement

    def _maybe_quantize_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply the paper's single static lattice-quantization level."""

        if self.q_level <= 0:
            return probs

        original_shape = probs.shape
        rows = _safe_normalize_probs(probs.reshape(-1, original_shape[-1]).float())
        quantized = torch.stack(
            [
                lattice_based_quantization_torch(row, self.q_level)
                for row in rows
            ],
            dim=0,
        )
        return _safe_normalize_probs(quantized).reshape(original_shape)


def get_tree_attn_self_mask(k_config: Tuple[int]):
    k_config = torch.tensor(k_config, dtype=torch.int)
    prod_size = torch.cumprod(k_config, dim=0)
    mask_size = prod_size.sum().item()
    attn_mask = torch.zeros((mask_size, mask_size), dtype=torch.bool)
    attn_mask = attn_mask.diagonal_scatter(torch.ones(mask_size))
    idx_queue = [(0, None, idx) for idx in range(k_config[0])]
    while len(idx_queue) != 0:
        depth, parent, idx = idx_queue.pop(0)
        if parent is not None:
            attn_mask[idx, : parent + 1] = attn_mask[parent, : parent + 1]

        if depth != len(k_config) - 1:
            idx_base = prod_size[:depth].sum().item()
            child_idx_base = prod_size[: depth + 1].sum().item()
            for child_idx_bias in range(k_config[depth + 1]):
                real_child_idx = (
                    (idx - idx_base) * k_config[depth + 1]
                    + child_idx_base
                    + child_idx_bias
                )
                idx_queue.append((depth + 1, idx, real_child_idx))
    return attn_mask


class TreeStrategy(Strategy):
    def __init__(
        self,
        draft_model,
        target_model,
        k_config: Tuple[int, ...],
        draft_model_temp: float = 1,
        target_model_temp: float = 1,
        q_level: int = 0,
    ) -> None:
        super().__init__(
            draft_model,
            target_model,
            k_config,
            draft_model_temp,
            target_model_temp,
            q_level,
        )

        prod_size = torch.cumprod(torch.tensor(k_config, dtype=torch.int), dim=0)
        prod_size = torch.cat((torch.zeros(1).to(prod_size), prod_size)).tolist()
        self.prod_size = prod_size
        self.cumulative_prod_size = torch.cumsum(
            torch.tensor(prod_size), dim=0
        ).tolist()

        self.tree_attn_self_mask = get_tree_attn_self_mask(k_config).to(
            device=self.draft_model_device
        )

    def _build_tree_step_mask(
        self,
        *,
        step: int,
        context_input_length: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if int(step) == 0:
            return None, None

        step_tree_attn_self_mask = self.tree_attn_self_mask[
            self.cumulative_prod_size[step - 1] : self.cumulative_prod_size[step],
            : self.cumulative_prod_size[step],
        ]
        width = int(self.prod_size[step])
        position_ids = torch.full(
            (1, width),
            int(context_input_length) + int(step) - 1,
            dtype=torch.long,
            device=self.draft_model_device,
        )
        context_attn_mask = torch.ones(
            (width, int(context_input_length)),
            dtype=torch.bool,
            device=self.draft_model_device,
        )
        step_tree_attn_mask = torch.cat((context_attn_mask, step_tree_attn_self_mask), dim=1)
        return step_tree_attn_mask, position_ids

    def _build_prefix_tree_mask(
        self,
        *,
        split_depth: int,
        context_input_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prefix_tree_len = int(self.cumulative_prod_size[split_depth])
        prefix_tree_attn_self_mask = self.tree_attn_self_mask[
            :prefix_tree_len,
            :prefix_tree_len,
        ]
        position_ids = (
            prefix_tree_attn_self_mask.long().sum(dim=1)
            + int(context_input_length)
            - 1
        ).unsqueeze(0)
        context_attn_mask = torch.ones(
            (prefix_tree_len, int(context_input_length)),
            dtype=torch.bool,
            device=self.draft_model_device,
        )
        tree_attn_mask = torch.cat(
            (context_attn_mask, prefix_tree_attn_self_mask),
            dim=1,
        )
        return tree_attn_mask, position_ids

    def _sample_tree_step(
        self,
        *,
        logits: torch.Tensor,
        step_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.draft_model_temp == 0:
            topk_logit, topk_index = logits.topk(k=step_k, dim=-1)
            topk_probs = torch.softmax(topk_logit, dim=-1)
            step_cand_probs = torch.zeros_like(logits)
            step_cand_probs.scatter_(dim=1, index=topk_index, src=topk_probs)
            return step_cand_probs, topk_index.view(1, -1)

        step_cand_probs = _softmax_probs(logits, self.draft_model_temp)
        step_cand_probs = self._maybe_quantize_probs(step_cand_probs)

        cand_tokens = torch.multinomial(
            step_cand_probs,
            step_k,
            replacement=False,
        ).view(1, -1)
        return step_cand_probs, cand_tokens

    def _empty_draft_output(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
    ) -> DecoderOnlyDraftOutput:
        return DecoderOnlyDraftOutput(
            sequences=input_ids.to(self.draft_model_device),
            past_key_values=past_key_values,
            cand_probs=tuple(),
            output_token_count=0,
        )

    def _generate_draft_range(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
        *,
        start_step: int,
        end_step: int,
        context_input_length: Optional[int] = None,
    ) -> DecoderOnlyDraftOutput:
        if end_step < start_step:
            return self._empty_draft_output(input_ids, past_key_values)

        base_input_ids = input_ids.to(self.draft_model_device)
        cand_probs: List[torch.Tensor] = []
        generated_chunks: List[torch.Tensor] = []
        init_input_length = (
            int(context_input_length)
            if context_input_length is not None
            else int(base_input_ids.size(1))
        )
        if past_key_values is not None:
            pruned_input_ids = base_input_ids[:, past_key_values[0][0].size(2) :]
        else:
            pruned_input_ids = base_input_ids

        output_token_count = 0

        for step in range(int(start_step), int(end_step) + 1):
            step_k = self.k_config[step]
            step_tree_attn_mask, position_ids = self._build_tree_step_mask(
                step=step,
                context_input_length=init_input_length,
            )
            outputs: BaseModelOutputWithPast = self.draft_model.model(
                input_ids=pruned_input_ids,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                tree_attn_mask=step_tree_attn_mask,
                position_ids=position_ids,
            )

            hidden_states = outputs.last_hidden_state
            if step == 0:
                hidden_states = hidden_states[0, -1:]
            else:
                hidden_states = hidden_states[0]
            logits = self.draft_model.lm_head(hidden_states)
            past_key_values = list(outputs.past_key_values)

            step_cand_probs, cand_tokens = self._sample_tree_step(
                logits=logits,
                step_k=step_k,
            )
            cand_probs.append(step_cand_probs.to(torch.float16))
            pruned_input_ids = cand_tokens
            generated_chunks.append(pruned_input_ids)
            output_token_count += int(cand_tokens.numel())
        sequences = (
            torch.cat([base_input_ids] + generated_chunks, dim=1)
            if generated_chunks
            else base_input_ids
        )
        return DecoderOnlyDraftOutput(
            sequences=sequences,
            past_key_values=past_key_values,
            cand_probs=tuple(cand_probs),
            output_token_count=int(output_token_count),
        )

    def generate_draft(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
    ) -> DecoderOnlyDraftOutput:
        return self._generate_draft_range(
            input_ids=input_ids,
            past_key_values=past_key_values,
            start_step=0,
            end_step=self.max_draft_len - 1,
        )

    def generate_prefix_draft(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
        split_depth: int,
    ) -> DecoderOnlyDraftOutput:
        split_depth = int(split_depth)
        if split_depth <= 0:
            return self._empty_draft_output(input_ids, past_key_values)
        if split_depth >= self.max_draft_len:
            return self.generate_draft(input_ids, past_key_values)
        return self._generate_draft_range(
            input_ids=input_ids,
            past_key_values=past_key_values,
            start_step=0,
            end_step=split_depth - 1,
        )

    def replay_prefix_on_cloud(
        self,
        prefix_output: DecoderOnlyDraftOutput,
        split_depth: int,
        base_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> DecoderOnlyDraftReplayOutput:
        split_depth = int(split_depth)
        if split_depth <= 0 or split_depth >= self.max_draft_len:
            return DecoderOnlyDraftReplayOutput(
                past_key_values=prefix_output.past_key_values,
                next_logits=None,
            )
        init_input_length = int(prefix_output.sequences.size(1) - (prefix_output.output_token_count or 0))
        full_input_ids = prefix_output.sequences.to(self.draft_model_device)
        base_input_ids = full_input_ids[:, :init_input_length]
        replay_tree_tokens = full_input_ids[:, init_input_length:]

        past_key_values = base_past_key_values
        logits = None

        if past_key_values is not None:
            context_input_ids = base_input_ids[:, past_key_values[0][0].size(2) :]
        else:
            context_input_ids = base_input_ids

        if context_input_ids.numel() > 0:
            outputs: BaseModelOutputWithPast = self.draft_model.model(
                input_ids=context_input_ids,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            past_key_values = list(outputs.past_key_values)

        expected_prefix_nodes = int(self.cumulative_prod_size[split_depth])
        if replay_tree_tokens.size(1) != expected_prefix_nodes:
            raise ValueError(
                "prefix replay tokens are incomplete: "
                f"expected {expected_prefix_nodes}, got {replay_tree_tokens.size(1)}"
            )

        tree_attn_mask, position_ids = self._build_prefix_tree_mask(
            split_depth=split_depth,
            context_input_length=init_input_length,
        )
        outputs = self.draft_model.model(
            input_ids=replay_tree_tokens,
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
            tree_attn_mask=tree_attn_mask,
            position_ids=position_ids,
        )
        final_width = int(self.prod_size[split_depth])
        final_hidden_states = outputs.last_hidden_state[0, -final_width:]
        logits = self.draft_model.lm_head(final_hidden_states)
        past_key_values = list(outputs.past_key_values)

        return DecoderOnlyDraftReplayOutput(
            past_key_values=past_key_values,
            next_logits=logits,
        )

    def generate_suffix_draft(
        self,
        prefix_output: DecoderOnlyDraftOutput,
        replay_output: DecoderOnlyDraftReplayOutput,
        split_depth: int,
    ) -> DecoderOnlyDraftOutput:
        split_depth = int(split_depth)
        if split_depth >= self.max_draft_len:
            return self._empty_draft_output(prefix_output.sequences, replay_output.past_key_values)
        if replay_output.next_logits is None:
            raise ValueError("replay_output.next_logits is required to generate suffix draft.")

        base_input_ids = prefix_output.sequences.to(self.draft_model_device)
        past_key_values = replay_output.past_key_values
        init_input_length = int(base_input_ids.size(1) - (prefix_output.output_token_count or 0))

        cand_probs: List[torch.Tensor] = []
        generated_chunks: List[torch.Tensor] = []
        pruned_input_ids = None
        output_token_count = 0
        logits = replay_output.next_logits

        for step in range(split_depth, self.max_draft_len):
            step_k = self.k_config[step]
            if step > split_depth:
                step_tree_attn_mask, position_ids = self._build_tree_step_mask(
                    step=step,
                    context_input_length=init_input_length,
                )
                outputs: BaseModelOutputWithPast = self.draft_model.model(
                    input_ids=pruned_input_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    tree_attn_mask=step_tree_attn_mask,
                    position_ids=position_ids,
                )
                hidden_states = outputs.last_hidden_state[0]
                logits = self.draft_model.lm_head(hidden_states)
                past_key_values = list(outputs.past_key_values)

            step_cand_probs, cand_tokens = self._sample_tree_step(
                logits=logits,
                step_k=step_k,
            )
            cand_probs.append(step_cand_probs.to(torch.float16))
            pruned_input_ids = cand_tokens
            generated_chunks.append(pruned_input_ids)
            output_token_count += int(cand_tokens.numel())

        sequences = (
            torch.cat([base_input_ids] + generated_chunks, dim=1)
            if generated_chunks
            else base_input_ids
        )
        return DecoderOnlyDraftOutput(
            sequences=sequences,
            past_key_values=past_key_values,
            cand_probs=tuple(cand_probs),
            output_token_count=int(output_token_count),
        )

    def _forward_target_model(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
        *,
        tree_attn_len: int,
        tree_attn_self_mask: torch.Tensor,
    ):
        input_ids = input_ids.to(self.target_model_device)
        init_input_length = input_ids.size(1) - tree_attn_len
        init_forward = False

        if past_key_values is not None:
            pruned_input_ids = input_ids[:, past_key_values[0][0].size(2) :]
        else:
            pruned_input_ids = input_ids
            init_forward = True

        if init_forward:
            tree_attn_mask = torch.zeros(
                (input_ids.size(1), input_ids.size(1)),
                dtype=torch.bool,
                device=self.target_model_device,
            )
            mask_cond = torch.arange(
                tree_attn_mask.size(-1), device=self.target_model_device
            )
            tree_attn_mask.masked_fill_(
                mask_cond < (mask_cond + 1).view(tree_attn_mask.size(-1), 1), 1
            )
            tree_attn_mask[-tree_attn_len:, -tree_attn_len:] = tree_attn_self_mask
            position_ids = tree_attn_mask.sum(dim=1) - 1

        else:
            # KV omits the final context token.
            tree_attn_mask = torch.ones(
                (
                    tree_attn_len + 1,
                    input_ids.size(1),
                ),
                dtype=torch.bool,
                device=self.target_model_device,
            )

            tree_attn_mask[1:, init_input_length:] = tree_attn_self_mask
            tree_attn_mask[0, init_input_length:] = 0
            position_ids = tree_attn_mask.sum(dim=1) - 1

        outputs: BaseModelOutputWithPast = self.target_model.model(
            input_ids=pruned_input_ids,
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
            tree_attn_mask=tree_attn_mask,
            position_ids=position_ids,
        )
        hidden_states = outputs.last_hidden_state
        past_key_values = list(outputs.past_key_values)

        logits = self.target_model.lm_head(hidden_states[:, -tree_attn_len - 1 :])
        return logits, past_key_values

    def verify(
        self,
        input_ids: torch.LongTensor,
        target_model_past_key_values,
        draft_model_past_key_values,
        cand_probs: Tuple[torch.FloatTensor, ...],
    ) -> DecoderOnlyVerificationOutput:
        """Verify one candidate tree and retain only the accepted path."""

        input_ids = input_ids.to(self.target_model_device)
        active_depth = len(cand_probs)
        if active_depth == 0:
            raise ValueError("cand_probs must contain at least one tree depth")

        tree_length = int(self.cumulative_prod_size[active_depth])
        tree_mask = self.tree_attn_self_mask[:tree_length, :tree_length]
        logits, target_model_past_key_values = self._forward_target_model(
            input_ids,
            target_model_past_key_values,
            tree_attn_len=tree_length,
            tree_attn_self_mask=tree_mask,
        )
        logits = logits[0]
        tree_tokens = input_ids[0, -tree_length:]
        context_length = input_ids.size(1) - tree_length

        if self.target_model_temp == 0:
            top_tokens = logits.argmax(dim=-1, keepdim=True)
            target_probs = torch.zeros_like(logits)
            target_probs.scatter_(1, top_tokens, 1.0)
        else:
            target_probs = _softmax_probs(logits, self.target_model_temp)

        current_target = target_probs[0]
        target_probs = target_probs[1:]
        accepted_tree_indices: List[int] = []
        group_offset = 0
        probability_index = 0

        accepted_depth = 0
        for depth in range(active_depth):
            level_start = self.cumulative_prod_size[depth] + group_offset
            accepted_branch = self.acceptance_check(
                current_target,
                cand_probs[depth][probability_index].float(),
                tree_tokens[level_start : level_start + self.k_config[depth]],
            )
            if accepted_branch is None:
                break

            accepted_depth = depth + 1
            tree_index = level_start + int(accepted_branch)
            accepted_tree_indices.append(tree_index)
            current_target = target_probs[tree_index]
            if accepted_depth < active_depth:
                probability_index = group_offset + int(accepted_branch)
                group_offset = probability_index * self.k_config[depth + 1]

        context_indices = torch.arange(
            context_length,
            device=self.target_model_device,
            dtype=torch.long,
        )
        if accepted_tree_indices:
            tree_indices = torch.tensor(
                accepted_tree_indices,
                device=self.target_model_device,
                dtype=torch.long,
            )
            keep_indices = torch.cat(
                (context_indices, context_length + tree_indices),
                dim=0,
            )
        else:
            tree_indices = None
            keep_indices = context_indices

        # Draft KV omits the deepest accepted token.
        if accepted_depth == active_depth and tree_indices is not None:
            if tree_indices.numel() > 1:
                draft_keep_indices = torch.cat(
                    (context_indices, context_length + tree_indices[:-1]),
                    dim=0,
                )
            else:
                draft_keep_indices = context_indices
        else:
            draft_keep_indices = keep_indices

        endpoint = torch.multinomial(
            _safe_normalize_probs(current_target),
            num_samples=1,
        ).to(input_ids.device)
        sequences = input_ids.index_select(1, keep_indices)
        sequences = torch.cat((sequences, endpoint[None]), dim=1)

        target_indices_by_device = {}
        for layer_index, (key, value) in enumerate(target_model_past_key_values):
            indices = target_indices_by_device.setdefault(
                key.device,
                keep_indices.to(key.device),
            )
            target_model_past_key_values[layer_index] = (
                key.index_select(2, indices),
                value.index_select(2, indices),
            )

        draft_indices_by_device = {}
        for layer_index, (key, value) in enumerate(draft_model_past_key_values):
            indices = draft_indices_by_device.setdefault(
                key.device,
                draft_keep_indices.to(key.device),
            )
            draft_model_past_key_values[layer_index] = (
                key.index_select(2, indices),
                value.index_select(2, indices),
            )

        return DecoderOnlyVerificationOutput(
            sequences=sequences,
            target_model_past_key_values=target_model_past_key_values,
            draft_model_past_key_values=draft_model_past_key_values,
            acceptance_count=accepted_depth,
        )
