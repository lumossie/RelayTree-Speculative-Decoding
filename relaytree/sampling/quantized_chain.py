"""Static quantize-sample single-chain speculative-decoding baseline."""

from __future__ import annotations

import time

import torch

from relaytree.sampling.utils import (
    calculate_quantization_bits,
    max_fn,
    norm_logits,
    quantized_logits,
    sample,
)


try:
    from transformers.generation.candidate_generator import (
        _crop_past_key_values as _legacy_crop_fn,
    )
except ImportError:
    try:
        from transformers.generation.utils import (
            _crop_past_key_values as _legacy_crop_fn,
        )
    except ImportError:
        _legacy_crop_fn = None


def _crop_cache(model, cache, length: int):
    if cache is None:
        return None
    if hasattr(cache, "crop"):
        return cache.crop(length)
    if isinstance(cache, tuple) and _legacy_crop_fn is not None:
        return _legacy_crop_fn(model, cache, length)
    raise TypeError(f"Unsupported KV-cache type: {type(cache)!r}")


class SpecSamplingEnv:
    """QS baseline retained for the paper comparison."""

    def __init__(
        self,
        approx_model,
        target_model,
        max_len: int = 64,
        temperature: float = 1.0,
        device: str = "cuda",
    ) -> None:
        self.approx_model = approx_model
        self.target_model = target_model
        self.max_len = int(max_len)
        self.temperature = float(temperature)
        self.device = device
        self.vocab_size = int(approx_model.config.vocab_size)
        self.uplink_rate = 1e9

        self.prefix = None
        self.prifex_len = 0  # Runner API typo.
        self.approx_past_key_values = None
        self.target_past_key_values = None
        self.total_time = 0.0
        self.total_uplink_time = 0.0
        self.total_uplink_bits = 0
        self.total_draft_time = 0.0
        self.total_target_verify_time = 0.0
        self.times_count = 0

    def reset(self, input_prefix: torch.Tensor) -> None:
        self.prefix = input_prefix
        self.prifex_len = int(input_prefix.shape[1])
        self.approx_past_key_values = None
        self.target_past_key_values = None
        self.total_time = 0.0
        self.total_uplink_time = 0.0
        self.total_uplink_bits = 0
        self.total_draft_time = 0.0
        self.total_target_verify_time = 0.0
        self.times_count = 0

    def _bits_per_distribution(self, q_level: int) -> int:
        # QS uses 16-bit token IDs.
        return calculate_quantization_bits(q_level, self.vocab_size) + 16

    @torch.no_grad()
    def step(self, static_action=(4, 96)):
        gamma, q_level = (int(static_action[0]), int(static_action[1]))
        start = time.perf_counter()
        random_numbers = torch.rand(gamma, device=self.device)
        candidates = self.prefix
        prefix_length = int(candidates.shape[1])
        quantized_probs = torch.zeros(
            candidates.shape[0], gamma, self.vocab_size, device=self.device
        )

        draft_start = time.perf_counter()
        for index in range(gamma):
            if self.approx_past_key_values is None:
                model_input = candidates
            else:
                cached_length = self.approx_past_key_values[0][0].size(2)
                model_input = candidates[:, cached_length:]
            output = self.approx_model(
                model_input,
                past_key_values=self.approx_past_key_values,
                use_cache=True,
            )
            self.approx_past_key_values = output.past_key_values
            logits = output.logits[:, -1, :]
            quantized_probs[:, index, :] = quantized_logits(
                logits,
                q_level,
                self.temperature,
            )
            candidates = torch.cat(
                (candidates, sample(quantized_probs[:, index, :])),
                dim=1,
            )
        self.total_draft_time += time.perf_counter() - draft_start

        verify_start = time.perf_counter()
        if self.target_past_key_values is None:
            target_input = candidates
        else:
            cached_length = self.target_past_key_values[0][0].size(2)
            target_input = candidates[:, cached_length:]
        target_output = self.target_model(
            target_input,
            past_key_values=self.target_past_key_values,
            use_cache=True,
        )
        self.target_past_key_values = target_output.past_key_values
        self.total_target_verify_time += time.perf_counter() - verify_start

        logits = target_output.logits
        target_probs = torch.stack(
            [
                norm_logits(
                    logits[:, logits.shape[1] - gamma + index - 1, :],
                    self.temperature,
                )
                for index in range(gamma + 1)
            ],
            dim=1,
        )

        accepted_end = prefix_length - 1
        all_accepted = True
        for index in range(gamma):
            token = candidates[:, prefix_length + index]
            acceptance = target_probs[:, index, token] / quantized_probs[:, index, token]
            if random_numbers[index] <= torch.clamp(acceptance, max=1.0):
                accepted_end += 1
                continue

            residual = max_fn(
                target_probs[:, accepted_end - prefix_length + 1, :]
                - quantized_probs[:, accepted_end - prefix_length + 1, :]
            )
            next_token = sample(residual)
            cache_length = accepted_end + 1
            self.target_past_key_values = _crop_cache(
                self.target_model,
                self.target_past_key_values,
                cache_length,
            )
            self.approx_past_key_values = _crop_cache(
                self.approx_model,
                self.approx_past_key_values,
                cache_length,
            )
            all_accepted = False
            break

        self.prefix = candidates[:, : accepted_end + 1]
        if all_accepted:
            next_token = sample(target_probs[:, -1, :])
        self.prefix = torch.cat((self.prefix, next_token), dim=1)

        elapsed = time.perf_counter() - start
        uplink_bits = gamma * self._bits_per_distribution(q_level)
        self.total_time += elapsed
        self.total_uplink_bits += uplink_bits
        self.total_uplink_time += uplink_bits / self.uplink_rate
        self.times_count += 1
        done = (self.prefix.shape[1] - self.prifex_len) >= self.max_len
        return None, -(elapsed + uplink_bits / self.uplink_rate), bool(done)

    def get_metrics(self) -> dict:
        generated = int(self.prefix.shape[1] - self.prifex_len)
        return {
            "generated_tokens": generated,
            "iterations": self.times_count,
            "compute_time": self.total_time,
            "uplink_bits": self.total_uplink_bits,
            "draft_time": self.total_draft_time,
            "target_verify_time": self.total_target_verify_time,
        }
