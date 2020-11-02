## Event-Triggered Communication in Parallel Machine Learning

The primary objective of this repository is to introduce EventGraD -  a novel communication algorithm
based on event-triggered communication to reduce communication in parallel machine learning.
A preliminary paper with further details can be found 
[here](https://www.researchgate.net/publication/344419608_EventGraD_Event-Triggered_Communication_in_Parallel_Stochastic_Gradient_Descent).
Please see `/dmnist/event/` for the EventGraD code.

## PyTorch C++ API meets MPI

The secondary objective of this repository is to serve as a starting point to implement
parallel/distributed machine learning using PyTorch C++ (LibTorch) and MPI. Apart from 
EventGraD, other popular distributed algorithms such as AllReduce based training
(`/dmnist/cent/`) and [decentralized training with neighbors](http://papers.nips.cc/paper/7117-can-decentralized-algorithms-outperform-centralized-algorithms-a-case-study-for-decentralized-parallel-stochastic-gradient-descent.pdf)(`/dmnist/decent/`)
are covered. There is also an [active pull request](https://github.com/pytorch/examples/pull/809) to merge some of this code to the PyTorch examples repository. 

## Plan of Development

While EventGraD is currently implemented on the MNIST dataset, we are working to extend it
for training state-of-the-art models like ResNet-50 on state-of-the-art datasets like ImageNet 
and study its theoretical properties.
