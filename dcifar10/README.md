Dependencies
-------------------------------------

1. [PyTorch C++/Libtorch](https://pytorch.org/get-started/locally/)
2. MPI
3. OpenCV

Note: This code has been tested on Libtorch 1.7.1 pre-cxx11 ABI without CUDA on Linux, 
Open MPI 4.0.1 compiled with gcc 8.3.0 and OpenCV 3.4

Dataset
------------------------------------

Please set the datasetPath variable in common/custom.hpp to the CIFAR-10 dataset comprising of raw images downloaded from [here](https://github.com/YoongiKim/CIFAR-10-images) 

Compiling
-------------------------------------

For compiling, we use CMake

```sh
cd event
mkdir build; cd build
cmake -DCMAKE_PREFIX_PATH=\path\to\libtorch ..
make
```

Running
-------------------------------------

```sh
mpirun -np ${NUM_PROCS} ./event ${ARGS}
```

${ARGS}[1] - Flag for file writing for debugging; 1 for enabling, 0 for disabling

${ARGS}[2] - Determines threshold type; 1 for adaptive threshold, 0 for non-adaptive threshold

The next arguments depends on the type of threshold:

### Adaptive threshold

${ARGS}[3] - Horizon parameter

Sample Run of adaptive threshold with horizon 1:

```sh
mpirun -np ${NUM_PROCS} ./event 0 1 1
```

### Non-adaptive (static) threshold

${ARGS}[3] - Value of static threshold

Sample Run of static threshold with constant 5e-4

```sh
mpirun -np ${NUM_PROCS} ./event 0 0 5e-4
```

Note: For comparison, choosing a horizon of 0 in the adaptive threshold or a constant of 0
in the non-adaptive threshold yields the algorithm without event-triggered communication as in Lian et. al (2017)

