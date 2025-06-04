# SOS-FGL

This project is the implementation code for **SOS-FGL** (Split, Overlap, and Share for Federated Graph Learning). This method aims to achieve efficient training and privacy protection of distributed graph neural networks through mechanisms such as graph partitioning and federated learning.

## Project Overview

- **SOS-FGL** combines multiple mechanisms including graph partitioning (e.g., Metis), federated learning (e.g., FedAvg), information bottleneck (IB), and adversarial attacks, supporting various common graph datasets (e.g., Cora, Citeseer, Pubmed, Polblogs).
- Clear code structure, easy to extend and reproduce.

## Main Files Description

- `main.py`: Project main entry, responsible for parameter parsing, data loading, training and testing processes.
- `fed_trainer.py`: Implementation of federated learning trainer.
- `trainer.py`: Implementation of centralized trainer.
- `clustering.py`: Implementation of graph partitioning and clustering.
- `util.py`: Common utility functions and data processing.
- `my_parser.py`: Command line parameter parsing.
- Other files are model, attack, and auxiliary tool modules.

## Environment Setup

Python 3.8 or higher is recommended. Main dependencies are:

- torch >= 2.1.0
- torch_geometric >= 2.6.1
- numpy >= 1.26.4
- pandas >= 2.2.3
- networkx >= 3.2.1
- texttable >= 1.7.0
- scipy >= 1.13.1
- scikit-learn >= 1.6.1
- matplotlib >= 3.9.2
- seaborn >= 0.13.2


