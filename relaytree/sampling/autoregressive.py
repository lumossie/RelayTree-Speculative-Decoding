"""Cloud-only autoregressive sampling baseline."""

from __future__ import annotations

import torch


@torch.no_grad()
def autoregressive_sampling(
    input_ids: torch.Tensor,
    model: torch.nn.Module,
    tokenizer,
    max_new_tokens: int,
    temperature: float = 1.0,
):
    """Generate tokens with the target model using the paper's AS settings."""
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "min_new_tokens": max_new_tokens,
    }
    if temperature <= 0:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
        )
    output = model.generate(input_ids, **generation_kwargs)
    return output[:, input_ids.shape[1] :], 0
