"""Relay-assisted tree speculative-decoding environment."""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import torch

from relaytree.inference.strategies import DecoderOnlyDraftOutput, TreeStrategy
from relaytree.sampling.utils import calculate_quantization_bits


def _crop_cache(cache, length: int):
    if cache is None:
        return None
    if hasattr(cache, "crop"):
        return cache.crop(length)
    cropped = []
    for layer in cache:
        key, value = layer[:2]
        key = key[:, :, :length, :]
        value = value[:, :, :length, :]
        cropped.append((key, value, *layer[2:]))
    return cropped


def _tree_token_count(k_config: Tuple[int, ...]) -> int:
    total = 0
    width = 1
    for branching in k_config:
        width *= branching
        total += width
    return total


def _tree_probability_count(k_config: Tuple[int, ...]) -> int:
    if not k_config:
        return 0
    total = 1
    width = 1
    for branching in k_config[:-1]:
        width *= branching
        total += width
    return total


def _prefix_token_count(k_config: Tuple[int, ...], split_depth: int) -> int:
    return _tree_token_count(k_config[:split_depth])


class TreeMCSDSpecSamplingEnv:
    """Run all-cloud, full-edge, or split-depth RelayTree decoding."""

    def __init__(
        self,
        draft_model,
        target_model,
        k_config: Tuple[int, ...] = (3, 1, 1),
        max_len: int = 64,
        temperature: float = 1.0,
        q_level: int = 96,
        scheme: str = "edge_full",
        relay_split_depth: int = -1,
    ) -> None:
        self.draft_model = draft_model
        self.target_model = target_model
        self.k_config = tuple(int(value) for value in k_config)
        self.max_len = int(max_len)
        self.temperature = float(temperature)
        self.q_level = int(q_level)
        self.scheme = str(scheme)
        if self.scheme not in {"cloud_all", "edge_full", "relay"}:
            raise ValueError(f"Unknown scheme: {scheme!r}")

        if relay_split_depth < 0:
            relay_split_depth = max(1, len(self.k_config) - 1)
        self.relay_split_depth = int(relay_split_depth)
        if not 0 <= self.relay_split_depth <= len(self.k_config):
            raise ValueError("relay_split_depth must lie within the tree depth")

        self.strategy = TreeStrategy(
            draft_model=draft_model,
            target_model=target_model,
            k_config=self.k_config,
            draft_model_temp=self.temperature,
            target_model_temp=self.temperature,
            q_level=self.q_level,
        )
        self.vocab_size = int(draft_model.config.vocab_size)
        self.token_bits = math.ceil(math.log2(self.vocab_size))
        self.uplink_rate = 1e6

        self.prefix: Optional[torch.Tensor] = None
        self.prefix_len = 0
        self.prifex_len = 0  # Runner API typo.
        self.draft_model_past_key_values = None
        self.target_model_past_key_values = None
        self._reset_metrics()

    def _reset_metrics(self) -> None:
        self.total_time = 0.0
        self.total_uplink_time = 0.0
        self.total_uplink_bits = 0
        self.total_edge_prefix_time = 0.0
        self.total_cloud_replay_time = 0.0
        self.total_cloud_suffix_time = 0.0
        self.total_target_verify_time = 0.0
        self.iterations = 0

    def _synchronize(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _timed(self, operation):
        self._synchronize()
        start = time.perf_counter()
        result = operation()
        self._synchronize()
        return result, time.perf_counter() - start

    def _set_strategy_q(self, q_level: int) -> None:
        self.strategy.q_level = int(q_level)

    def _uplink_bits_per_iteration(self) -> int:
        if self.scheme == "cloud_all":
            return 0
        if self.scheme == "relay":
            return _prefix_token_count(self.k_config, self.relay_split_depth) * self.token_bits
        token_bits = _tree_token_count(self.k_config) * self.token_bits
        probability_bits = 0
        if self.temperature > 0 and self.q_level > 0:
            probability_bits = _tree_probability_count(self.k_config) * int(
                calculate_quantization_bits(self.q_level, self.vocab_size)
            )
        return token_bits + probability_bits

    @torch.no_grad()
    def reset(self, input_prefix: torch.Tensor, prefill: bool = True) -> None:
        self.prefix = input_prefix
        self.prefix_len = int(input_prefix.shape[1])
        self.prifex_len = self.prefix_len
        self.draft_model_past_key_values = None
        self.target_model_past_key_values = None
        self._reset_metrics()

        if not prefill or self.prefix_len <= 1:
            return
        prefill_ids = input_prefix[:, :-1]
        draft_output = self.draft_model.model(
            input_ids=prefill_ids.to(self.strategy.draft_model_device),
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        target_output = self.target_model.model(
            input_ids=prefill_ids.to(self.strategy.target_model_device),
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        self.draft_model_past_key_values = list(draft_output.past_key_values)
        self.target_model_past_key_values = list(target_output.past_key_values)

    def _generate_draft(self):
        if self.scheme == "edge_full":
            self._set_strategy_q(self.q_level)
            output, elapsed = self._timed(
                lambda: self.strategy.generate_draft(
                    self.prefix,
                    self.draft_model_past_key_values,
                )
            )
            self.total_edge_prefix_time += elapsed
            return output, elapsed

        if self.scheme == "cloud_all":
            self._set_strategy_q(0)
            output, elapsed = self._timed(
                lambda: self.strategy.generate_draft(
                    self.prefix,
                    self.draft_model_past_key_values,
                )
            )
            self.total_cloud_suffix_time += elapsed
            return output, elapsed

        self._set_strategy_q(0)
        prefix_output, edge_time = self._timed(
            lambda: self.strategy.generate_prefix_draft(
                self.prefix,
                self.draft_model_past_key_values,
                self.relay_split_depth,
            )
        )
        replay_output, replay_time = self._timed(
            lambda: self.strategy.replay_prefix_on_cloud(
                prefix_output,
                self.relay_split_depth,
                self.draft_model_past_key_values,
            )
        )
        suffix_output, suffix_time = self._timed(
            lambda: self.strategy.generate_suffix_draft(
                prefix_output,
                replay_output,
                self.relay_split_depth,
            )
        )
        self.total_edge_prefix_time += edge_time
        self.total_cloud_replay_time += replay_time
        self.total_cloud_suffix_time += suffix_time
        return (
            DecoderOnlyDraftOutput(
                sequences=suffix_output.sequences,
                past_key_values=suffix_output.past_key_values,
                cand_probs=tuple(prefix_output.cand_probs) + tuple(suffix_output.cand_probs),
            ),
            edge_time + replay_time + suffix_time,
        )

    @torch.no_grad()
    def step(self):
        if self.prefix is None:
            raise RuntimeError("reset() must be called before step()")
        if self.prefix.shape[1] - self.prefix_len >= self.max_len:
            return None, 0.0, True

        previous_length = int(self.prefix.shape[1])
        draft_output, draft_time = self._generate_draft()
        self.draft_model_past_key_values = draft_output.past_key_values

        verification, verify_time = self._timed(
            lambda: self.strategy.verify(
                draft_output.sequences,
                self.target_model_past_key_values,
                draft_output.past_key_values,
                draft_output.cand_probs,
            )
        )
        self.total_target_verify_time += verify_time
        self.prefix = verification.sequences[:, : self.prefix_len + self.max_len]
        cache_length = max(int(self.prefix.shape[1]) - 1, 0)
        self.draft_model_past_key_values = _crop_cache(
            verification.draft_model_past_key_values,
            cache_length,
        )
        self.target_model_past_key_values = _crop_cache(
            verification.target_model_past_key_values,
            cache_length,
        )

        uplink_bits = self._uplink_bits_per_iteration()
        uplink_time = uplink_bits / self.uplink_rate
        self.total_uplink_bits += uplink_bits
        self.total_uplink_time += uplink_time
        self.total_time += draft_time + verify_time
        self.iterations += 1

        generated = int(self.prefix.shape[1]) - self.prefix_len
        done = generated >= self.max_len
        emitted = int(self.prefix.shape[1]) - previous_length
        return None, -(draft_time + verify_time + uplink_time), bool(done or emitted <= 0)

    def get_metrics(self) -> dict:
        generated = 0 if self.prefix is None else int(self.prefix.shape[1] - self.prefix_len)
        wall_time = self.total_time + self.total_uplink_time
        return {
            "generated_tokens": generated,
            "iterations": self.iterations,
            "throughput": generated / wall_time if wall_time > 0 else 0.0,
            "compute_time": self.total_time,
            "uplink_time": self.total_uplink_time,
            "uplink_bits": self.total_uplink_bits,
            "edge_prefix_time": self.total_edge_prefix_time,
            "cloud_replay_time": self.total_cloud_replay_time,
            "cloud_suffix_time": self.total_cloud_suffix_time,
            "target_verify_time": self.total_target_verify_time,
        }
