# QIRO_Routing
This repository contains the code for implementing [QIRO](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.020327) for the Travelling Salesperson Problem (TSP) and the Capacitated Vehicle Routing Problem (CVRP). This work introduces problem-specific update classical update steps for QIRO as well as update steps on a qubit-level. As quantum algorithms, QAOA, VQE, simulated annealing and an exact solver are imnplemented.

### Installation

Set up a conda environment for QIRO:

```
conda create -n qiro python=3.9.18
```
Activate the environment:
```
conda activate qiro
```
Install the requirements:
```
pip install -r requirements.txt
```

### Usage
[This notebook](https://github.com/Minidimi/QIRO_Routing/blob/main/test_runs.ipynb) allows for testing the default instances. The quantum algorithm can be chosen by entering it as a parameter for QIRO while the update steps are implemented in different functions. When using QIRO in other contexts, [this file](https://github.com/Minidimi/QIRO_Routing/blob/main/src/QIRO_Optimizer.py) offers the necessary implementations. 
