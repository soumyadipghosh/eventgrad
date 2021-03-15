## Event-Triggered Communication in Parallel Machine Learning

The primary objective of this repository is to introduce EventGraD -  a novel communication algorithm
based on event-triggered communication to reduce communication in parallel machine learning.
A preliminary paper with further details can be found 
[here](https://www.researchgate.net/publication/344419608_EventGraD_Event-Triggered_Communication_in_Parallel_Stochastic_Gradient_Descent). An extended version with mathematical formulations, theoretical proofs of convergence and newer results can be found [here](https://arxiv.org/abs/2103.07454). Please see `/dmnist/event/` for the EventGraD code on MNIST and `/dcifar10/event` for the EventGraD code on CIFAR-10.

## PyTorch C++ API meets MPI

The secondary objective of this repository is to serve as a starting point to implement
parallel/distributed machine learning using PyTorch C++ (LibTorch) and MPI. Apart from 
EventGraD, other popular distributed algorithms such as AllReduce based training
(`/dmnist/cent/`) and [decentralized training with neighbors](http://papers.nips.cc/paper/7117-can-decentralized-algorithms-outperform-centralized-algorithms-a-case-study-for-decentralized-parallel-stochastic-gradient-descent.pdf)(`/dmnist/decent/`)
are covered. The AllReduce based training code was contributed to the pytorch/examples 
repository [here](https://github.com/pytorch/examples/tree/master/cpp/distributed) through [this pull request](https://github.com/pytorch/examples/pull/809). 
