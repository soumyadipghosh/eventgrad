## Event-Triggered Communication in Parallel Machine Learning

The primary objective of this repository is to introduce EventGraD - a novel communication algorithm
based on event-triggered communication to reduce communication in parallel machine learning. EventGraD considers the decentralized setting where communication happens only with the neighbor processors at every iteration instead of an AllReduce involving every processor at every iteration. The main idea is to trigger communication in events only when the parameter to be communicated changes by a threshold. For details on how to choose an adaptive threshold and convergence proofs, please refer to the publications. EventGraD saves around 70% of the messages in MNIST and 60% of the messages on CIFAR-10. Please see `/dmnist/event/` for the EventGraD code on MNIST and `/dcifar10/event` for the EventGraD code on CIFAR-10.

## PyTorch C++ API meets MPI

The secondary objective of this repository is to serve as a starting point to implement
parallel/distributed machine learning using PyTorch C++ (LibTorch) and MPI. Apart from 
EventGraD, other popular distributed algorithms such as AllReduce based training
(`/dmnist/cent/`) and [decentralized training with neighbors](http://papers.nips.cc/paper/7117-can-decentralized-algorithms-outperform-centralized-algorithms-a-case-study-for-decentralized-parallel-stochastic-gradient-descent.pdf)(`/dmnist/decent/`)
are covered. The AllReduce based training code was contributed to the pytorch/examples 
repository [here](https://github.com/pytorch/examples/tree/master/cpp/distributed) through [this pull request](https://github.com/pytorch/examples/pull/809).

## Publications

1. Soumyadip Ghosh, Bernardo Aquino and Vijay Gupta, "EventGraD: Event-Triggered Communication in Parallel Machine Learning", accepted, to appear in Elsevier Neurocomputing, [arXiv:2103.07454](https://arxiv.org/abs/2103.07454)

2. Soumyadip Ghosh and Vijay Gupta, "EventGraD: Event-Triggered Communication in Parallel Stochastic Gradient Descent", [Workshop on Machine Learning in HPC Environments (MLHPC), Supercomputing Conference (SC), Virtual, November 2020](https://www.researchgate.net/publication/344419608_EventGraD_Event-Triggered_Communication_in_Parallel_Stochastic_Gradient_Descent) 
