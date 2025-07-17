# Ensemble Model Predictive Safety Certification `x_mpsc`

This is the repository containing the code and experiments of the X-MPSC Paper
[(see PDF)](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p724.pdf)

x-model predictive safety certification. X is a placeholder for high-capacity probabilistic dynamics models such as the ensemble of probabilistic NNets in PETS [1].



## Installation

First, download `x_mpsc` and change the directory
```
git clone https://github.com/SvenGronauer/x-mpsc
cd ./x_mpsc
```

Then, create virtual environment via
```
python3 -m venv .venv
```

Activate the venv
```
source .venv/bin/activate
```

Install requirements
```
pip3 install -r requirements.txt
```

Finally, install `x_mpsc` as an editable repository:
```
pip install -e .
```

> Note: the implemented algorithms usually make use of `mpi4py` which is 
> on OpenMPI framework that can be used for multi-processing. However, its 
> installation is not a mandotory and 1 CPU core is used by default.


## Training

To train an agent with the Lag-TRPO algorithm call:
```bash
python -m x_mpsc.train --alg lag-trpo --env SimplePendulum-v0
```

To run an RL training with MBPO+X-MPSC, execute:
```bash
python -m x_mpsc.train --alg mbpo --env SimplePendulum-v0 --use_mpsc True
```

To use X-MPSC with a prior model, then run:
```bash
python -m x_mpsc.train --alg mbpo --env SimplePendulum-v0 --use_prior_model True --use_mpsc True
```


## Visualizing 

After an RL model has been trained and its checkpoint has been saved on your 
disk, you can visualize the checkpoint:
```
$ python -m x_mpsc.play --ckpt PATH_TO_CKPT
```
where PATH_TO_CKPT is the path to the checkpoint, e.g.
`/var/tmp/user/SimplePendulum-v0/mbpo/2023-05-17__09-28-32/seed_33312`

You can also display a random policy via:
```bash
python -m x_mpsc.play --random --env SimplePendulum-v0
```

## Paper Experiments

The experiments run files of the four tested environments
1) Simple Pendulum
2) Cart Pole  
3) Two-Link-Arm and
4) The Drone

can be found in the respective subfolder in the ``experiments`` directory. 
To run the MBPO+X-MPSC experiments, execute e.g.
```
$ python experiments/simple_pendulum/run_mpbo_experiments.py
```


## Structure of this repo

    .
    ├── experiments         (Scripts used to run paper experiments)
    ├── x_mpsc
    │   ├── algs            (algorithm implementations)
    │   ├── common          (utility and helper files)
    │   ├── envs            (environments based on Gym-API)
    │   ├── models          (symbolic models / Neural networks descriptions the dynamical systems)
    │   └── mpc             (Classes for MPC solvers and helper functions)
    │   └── mpsc            (Classes for MPSC solvers and helper functions)
    ├── ...
    └── Readme.md



# Implemented Algorithms

+ Constrained Policy Optimization (CPO)
+ Lagrangian Trust-region Policy Optimization (Lag-TRPO)
+ Lyapunov Barrier Policy Optimization (LBPO)
+ Model-based Policy Optimization (MBPO)
+ Probabilistic Ensembles with Trajectory Sampling (PETS)
+ Proximal Policy Optimization (PPO)
+ Soft-actor Critic (SAC)
+ Safety Q-function for Reinforcement Learning (SQRL)
+ Trust-region Policy Optimization (TRPO)


# Paper Citation

```
@inproceedings{Gronauer:2024:xmpsc,
	author = {Gronauer, Sven and Haider, Tom and Schmoeller da Roza, Felippe and Diepold, Klaus},
	booktitle = {23rd International Conference on Autonomous Agents and Multiagent Systems},
	series = {AAMAS},
	title = {Reinforcement Learning with Ensemble Model Predictive Safety Certification},
	year = {2024},
	url = {https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p724.pdf}
}
```

# References 


[1] S. Gronauer, T. Haider, F. Schmoeller da Roza, and K. Diepold. “Reinforcement Learning with Ensemble Model Predictive Safety Certification”. In: Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems (AAMAS). 2024

[2] Chua et al.; “Deep reinforcement learning in a handful of trials using probabilistic dynamics models”; 2018
