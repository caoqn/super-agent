# MASBench Reasoning attribution

This adapter includes code derived from the official [MASBench repository](https://github.com/BUPT-GAMMA/MASBench). The upstream source is licensed under MIT; see the accompanying [`LICENSE-MASBench`](LICENSE-MASBench).

The upstream Reasoning task is adapted from [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://hotpotqa.github.io/) by Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning (EMNLP 2018). HotpotQA is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The Reasoning task rewrites questions and context passages and distributes evidence across two agents. This repository does not include the full `reasoning.json` dataset; obtain task data from the official benchmark source.

For this repository's adapter dependencies and run commands, see the [AgentNet × MASBench instructions](README.md), the [MASBench data notes](../../README.md), and the [repository setup guide](../../../../README.md).
