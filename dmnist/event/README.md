Instructions 
-------------------------------------

Dependencies
-------------------------------------

1. Libtorch 
2. MPI

Dataset
------------------------------------

Please set the filename variable to the MNIST dataset path

Compiling
-------------------------------------

For compiling, we use CMake 

```sh
mkdir build; cd build
cmake -DCMAKE_PREFIX_PATH=\path\to\libtorch ..
make
```

Note that a C++11 compliant compiler is required.

Running
-------------------------------------

```sh
mpirun -np ${NUM_PROCS} ./event ${ARGS}
```

${ARGS}[1] - Flag for file writing for debugging; 1 for enabling, 0 for disabling
${ARGS}[2] - Determines threshold type; 1 for adaptive threshold, 0 for non-adaptive threshold 

The next arguments depends on the type of threshold:

## Adaptive threshold

${ARGS}[3] - Horizon parameter

Sample Run of adaptive threshold with horizon 1:

```sh
mpirun -np ${NUM_PROCS} ./event 0 1 1
```

## Non-adaptive (static) threshold

${ARGS}[3] - Value of static threshold

Sample Run of static threshold with constant 5e-4

```sh
mpirun -np ${NUM_PROCS} ./event 0 0 5e-4
```
