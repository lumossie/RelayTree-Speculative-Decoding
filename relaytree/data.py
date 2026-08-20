"""Dataset prompt formatting shared by all paper experiments."""

from __future__ import annotations

import os
from typing import Any


def count_parameters(model: Any) -> int:
    """Return the total model parameter count."""
    return sum(parameter.numel() for parameter in model.parameters())


def get_input(args: Any, sample: Any) -> str:
    """Format the exact prompts used for the three paper tasks."""
    model_id = os.path.basename(str(args.target_model_name).rstrip("/")).lower()
    is_supported = any(
        marker in model_id
        for marker in ("pythia", "gpt-neox", "gpt_neox", "qwen")
    )
    if not is_supported:
        raise ValueError(
            "The public release reproduces only the Pythia and Qwen model pairs "
            f"reported in the paper; received {args.target_model_name!r}."
        )

    if args.task == "summarize":
        article = sample.get("article", sample.get("document", ""))
        return f"Summarize the following article:\n\n{article}\n\nSummary:"

    if args.task == "translate":
        german = sample["translation"]["de"]
        return (
            "Translate the following German text to English:\n\n"
            f"{german}\n\nEnglish:"
        )

    if args.task == "alpaca":
        instruction = sample.get("instruction", "")
        extra_input = sample.get("input", "")
        if extra_input:
            return (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. Write a response that appropriately "
                "completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{extra_input}\n\n"
                "### Response:\n"
            )
        return (
            "Below is an instruction that describes a task. Write a response that "
            "appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            "### Response:\n"
        )

    raise ValueError(f"Unsupported task: {args.task!r}")
