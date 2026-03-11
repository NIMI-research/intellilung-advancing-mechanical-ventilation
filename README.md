# About

This repository contains the implementation of the experiments from the paper  
[Advancing Safe Mechanical Ventilation Using Offline RL With Hybrid Actions and Clinically Aligned Rewards](https://arxiv.org/abs/2506.14375).

## Structure

The repository is divided into two main modules:

- `algo_src`
- `data_pipelines`

The `algo_src` module contains the algorithms, along with the training and evaluation code.  
The `data_pipelines` module contains the data pipelines for all datasets, as well as data validation tests.

To run the algorithms or data pipelines, follow the instructions in the README of each module.

## Citation
If you use our work in your research, please cite:
```
@misc{yousuf2026advancingsafemechanicalventilation,
      title={Advancing Safe Mechanical Ventilation Using Offline RL With Hybrid Actions and Clinically Aligned Rewards}, 
      author={Muhammad Hamza Yousuf and Jason Li and Sahar Vahdati and Raphael Theilen and Jakob Wittenstein and Jens Lehmann},
      year={2026},
      eprint={2506.14375},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.14375}, 
}
```