"""Run the RelayTree reference implementation.

Use ``python run.py --help`` for model, task, tree, quantization, and relay
options. Defaults reproduce the main Pythia setting used in the paper.
"""

from relaytree.benchmark import main


if __name__ == "__main__":
    main()
