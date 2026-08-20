"""Minimal benchmark for autoregressive, QS, and RelayTree decoding."""

from __future__ import annotations

import argparse
import contextlib
import random
import time
from types import SimpleNamespace
from typing import Iterable, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

from relaytree.data import count_parameters, get_input
from relaytree.sampling.autoregressive import autoregressive_sampling
from relaytree.sampling.quantized_chain import SpecSamplingEnv
from relaytree.sampling.relay_tree import TreeMCSDSpecSamplingEnv


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RelayTree reference benchmark (AS, QS, and RelayTree)"
    )
    parser.add_argument(
        "--task",
        default="summarize",
        choices=("summarize", "translate", "alpaca", "custom"),
        help="dataset task; custom uses --input",
    )
    parser.add_argument("--input", default="Explain speculative decoding.")
    parser.add_argument("--approx_model_name", default="EleutherAI/pythia-1b")
    parser.add_argument("--target_model_name", default="EleutherAI/pythia-12b")
    parser.add_argument("--max_tokens", "-M", type=int, default=128)
    parser.add_argument("--max_prompt_tokens", type=int, default=1900)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_eval", type=int, default=128)
    parser.add_argument("--sample_start", type=int, default=1000)
    parser.add_argument("--gamma", "-g", type=int, default=4, help="QS candidate length")
    parser.add_argument("--q_level", type=int, default=96)
    parser.add_argument("--tree_k", default="3,1,1", help="branching by depth")
    parser.add_argument(
        "--scheme",
        default="edge_full",
        choices=("cloud_all", "edge_full", "relay"),
    )
    parser.add_argument(
        "--relay_split_depth",
        type=int,
        default=-1,
        help="prefix depth generated at the edge in relay mode",
    )
    parser.add_argument("--uplink_rate", type=float, default=5e6, help="uplink bits/s")
    parser.add_argument("--skip_qs", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--seed", type=int, default=1235)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def isolated_seed(seed: int):
    """Keep one method from consuming another method's random stream."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        seed_all(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def parse_tree_k(raw: str) -> Tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("--tree_k must contain positive comma-separated integers")
    return values


def model_dtype(model_name: str, local_files_only: bool):
    config = AutoConfig.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    dtype = getattr(config, "torch_dtype", None)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype, None)
    return dtype or torch.float16


def load_model(model_name: str, local_files_only: bool):
    from relaytree.models.loader import get_tree_attn_causallm_class

    model_class = get_tree_attn_causallm_class(
        model_name,
        local_files_only=local_files_only,
    )
    return model_class.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=model_dtype(model_name, local_files_only),
        local_files_only=local_files_only,
    )


def load_samples(args: argparse.Namespace) -> Iterable:
    if args.task == "custom":
        return [None]

    dataset_specs = {
        "summarize": ("abisee/cnn_dailymail", "3.0.0", "test"),
        "translate": ("wmt14", "de-en", "test"),
        "alpaca": ("yahma/alpaca-cleaned", None, "train"),
    }
    name, subset, split = dataset_specs[args.task]
    dataset = load_dataset(name, subset)[split] if subset else load_dataset(name)[split]
    start = int(args.sample_start)
    end = start + int(args.num_eval)
    if start < 0 or end > len(dataset):
        raise ValueError(f"dataset range [{start}, {end}) exceeds split size {len(dataset)}")
    return dataset.select(range(start, end))


def run_autoregressive(
    input_ids: torch.Tensor,
    model,
    tokenizer,
    args: argparse.Namespace,
) -> float:
    start = time.perf_counter()
    output, _ = autoregressive_sampling(
        input_ids,
        model,
        tokenizer,
        args.max_tokens,
        temperature=args.temperature,
    )
    elapsed = time.perf_counter() - start
    return float(output.shape[1]) / max(elapsed, 1e-9)


def run_environment(env, input_ids: torch.Tensor, step_kwargs: dict | None = None) -> float:
    start = time.perf_counter()
    env.reset(input_ids)
    reset_time = time.perf_counter() - start
    done = False
    while not done:
        _, _, done = env.step(**(step_kwargs or {}))

    prompt_length = int(getattr(env, "prifex_len", input_ids.shape[1]))
    generated = int(env.prefix.shape[1]) - prompt_length
    compute = float(getattr(env, "total_time", 0.0))
    uplink = float(getattr(env, "total_uplink_time", 0.0))
    return generated / max(reset_time + compute + uplink, 1e-9)


def main() -> None:
    args = parse_arguments()
    seed_all(args.seed)
    tree_k = parse_tree_k(args.tree_k)

    tokenizer = AutoTokenizer.from_pretrained(
        args.approx_model_name,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    draft_model = load_model(args.approx_model_name, args.local_files_only)
    target_model = load_model(args.target_model_name, args.local_files_only)
    for model in (draft_model, target_model):
        if int(model.config.vocab_size) != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))

    draft_parameters = count_parameters(draft_model)
    target_parameters = count_parameters(target_model)
    print(
        f"Loaded draft={draft_parameters / 1e9:.2f}B and "
        f"target={target_parameters / 1e9:.2f}B parameters"
    )

    qs_env = None
    if not args.skip_qs:
        qs_env = SpecSamplingEnv(
            approx_model=draft_model,
            target_model=target_model,
            max_len=args.max_tokens,
            temperature=args.temperature,
        )
        qs_env.uplink_rate = args.uplink_rate

    relay_env = TreeMCSDSpecSamplingEnv(
        draft_model=draft_model,
        target_model=target_model,
        k_config=tree_k,
        max_len=args.max_tokens,
        temperature=args.temperature,
        q_level=args.q_level,
        scheme=args.scheme,
        relay_split_depth=args.relay_split_depth,
    )
    relay_env.uplink_rate = args.uplink_rate

    scores = {"AS": [], "RelayTree": []}
    if qs_env is not None:
        scores["QS"] = []

    prompt_args = SimpleNamespace(task=args.task, target_model_name=args.target_model_name)
    for sample_index, sample in enumerate(load_samples(args)):
        prompt = args.input if args.task == "custom" else get_input(prompt_args, sample)
        input_ids = tokenizer(
            [prompt],
            return_tensors="pt",
        ).input_ids[:, : args.max_prompt_tokens]

        base_seed = args.seed + sample_index * 10
        with isolated_seed(base_seed):
            scores["AS"].append(
                run_autoregressive(input_ids, target_model, tokenizer, args)
            )
        if qs_env is not None:
            with isolated_seed(base_seed + 1):
                scores["QS"].append(
                    run_environment(
                        qs_env,
                        input_ids,
                        {"static_action": (args.gamma, args.q_level)},
                    )
                )
        with isolated_seed(base_seed + 2):
            scores["RelayTree"].append(run_environment(relay_env, input_ids))

        print(f"Completed sample {sample_index + 1}", flush=True)

    print("\nAverage throughput")
    for method, values in scores.items():
        print(f"{method:>9}: {float(np.mean(values)):.2f} tokens/s")


if __name__ == "__main__":
    main()
