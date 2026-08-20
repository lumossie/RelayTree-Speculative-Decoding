# RelayTree

PyTorch reference implementation of **“RelayTree: Relay-Assisted Tree-Based
Speculative Decoding for Efficient Edge-Cloud LLM Inference.”**

RelayTree lets an edge SLM generate a prefix subtree, sends only the prefix
tokens to the cloud, reconstructs the SLM states through cloud replay, and then
completes and verifies the draft tree with the cloud LLM.

## Structure

```text
run.py                    example and benchmark entry point
relaytree/sampling/       RelayTree, QS, and autoregressive decoding
relaytree/inference/      tree construction, replay, and verification
relaytree/models/         Pythia and Qwen2 Tree-Attention backends
plotresult/               final figures used in the paper
```

## Paper Figures

The repository includes only the three simulation-result PDF figures used in
the paper:

- `figure_03_throughput_comparison.pdf`
- `figure_04_temperature_sweep.pdf`
- `figure_05_relay_deployment.pdf`

## Installation

```bash
pip install -r requirements.txt
```

The implementation was tested with Python 3.11, PyTorch 2.7.0, Transformers
4.47.0, and an NVIDIA A100 GPU.

## Example

The default command uses Pythia-1B/Pythia-12B, tree shape `(3,1,1)`, `Q=96`,
temperature 1.0, and a 5 Mbps uplink:

```bash
python run.py --task custom --input "Explain speculative decoding." --num_eval 1
```

Run the relay scheme with a split after the first tree depth:

```bash
python run.py \
  --task custom \
  --input "Explain speculative decoding." \
  --num_eval 1 \
  --scheme relay \
  --tree_k 3,1,1 \
  --relay_split_depth 1 \
  --q_level 128
```

Use `python run.py --help` for other model, dataset, tree, and communication
options. The supported paper tasks are CNN/DailyMail summarization, WMT14
German-to-English translation, and Alpaca instruction following.

The benchmark prints average throughput in tokens per second. Generated result
files, downloaded models, and dataset caches are intentionally not included.

## Models

The code supports the model pairs evaluated in the paper:

- `EleutherAI/pythia-1b` → `EleutherAI/pythia-12b`
- `Qwen/Qwen2.5-0.5B` → `Qwen/Qwen2.5-14B`

Models and datasets are downloaded from Hugging Face and are not included in
this repository.

## Citation

```bibtex
@article{xue2026relaytree,
  title   = {RelayTree: Relay-Assisted Tree-Based Speculative Decoding for Efficient Edge-Cloud LLM Inference},
  author  = {Xue, Tingyue and Li, Hanlei and Zhang, Guangyi and Hou, Qiushuo and Cai, Yunlong and Yu, Guanding},
  journal = {Submitted for publication},
  year    = {2026}
}
```

## License

This project is released under the Apache-2.0 License.
