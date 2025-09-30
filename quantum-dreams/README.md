**World-models PPO Reinforcement Learning for Quantum Control with Physical Constraints**

## Installation

To set up the environment and install dependencies, follow these steps:

### Create and Activate a Virtual Environment

Using Conda:

```sh
export CONPREFIX=qiskit
conda create --prefix $CONPREFIX python=3.10 -y
conda activate $CONPREFIX
```

### Install Dependencies

Install JAX with CUDA support:

```sh
conda install -c nvidia cuda
pip install --upgrade "jax[cuda12]"
```

Install additional required packages:

```sh
pip install qiskit-dynamics gymnax evosax distrax optax flax numpy brax wandb flashbax diffrax
```

## Overview

The implementation is contained in the `rl_working` directory. Our PPO+world-models algorithm implementation is based on the JAX-based framework [PureJAX-RL](https://github.com/luchris429/purejax-rl). The other implementations follow the structure of [CleanRL](https://github.com/vwxyzjn/cleanrl). We provide the following RL implementations:

## Acknowledgement

This project is adapted from and inspired by the open-source repository 
[RL4qcWpc](https://github.com/jan-o-e/RL4qcWpc) by jan-o-e.  
We thank the author for sharing their work, which served as a valuable reference for our implementation.

## Experiment
First,train and collect data using ppo:
```sh
WANDB_MODE=offline python ppo.py --env multi_stirap --n_sections_multi 10
```

Then，train the LSTM using the collected data：
```sh
python trainlstm.py
```

Finally,train the PPO with world-models:
```sh
WANDB_MODE=offline python train-ppo-with-world-models.py --env multi_stirap --n_sections_multi 10
```
To view the final result:
```sh
python print-result-ppo.py
python print-result-wm.py
```

## Final Notes

Thank you for your interest in **Quantum Dreams**!  
We welcome all contributions — feel free to submit issues, feature requests, or pull requests.  


### Citation

```bibtex
@misc{ernst2025reinforcementlearningquantumcontrol,
      title={Quantum Dreams: Model-Based Reinforcement Learning of
Coherent Control}, 
      author={Yueyi Liu},
      year={2025},
}
