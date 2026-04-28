# Dense Systems of Linear Equations
### EC527 Final Project, Spring 2026
### Jiaxing Wang, Alvin Yan, Owen Jiang

### Summary
This project explores the implementation and optimization of dense systems of linear equations. We start with a naive implementation of solving Ax=b through LU decomposition and forward and backward passes. Then, we optimize LU decomposition through blocking, vectorization, and parallelization through OpenMP and CUDA.

### Implementations
#### Serial
1. Serial Naive
2. Serial Blocked, version 1
3. Serial Blocked, version 2
#### Vectorized
4. AVX
#### OpenMP
5. OpenMP Naive
6. OpenMP Alternate Access Pattern
7. OpenMP Attempts at Optimization (not good)
#### CUDA
8. CUDA Single-Kernel
9. CUDA Two Kernels, Global Memory
10. CUDA Two Kernels, Local Memory
#### The Benchmark
11. cuBLAS (Using cuBLAS library)

To implement in the future: CUDA with Shared Memory and Blocking

### Compilation Notes
To compile all tests:
nvcc -O1 -Xcompiler "-fopenmp,-mavx,-mfma" main.cu sole_serial.cpp sole_omp.cpp sole_gpu.cu -o test_sole -lcusolver -lcublas -lm 

To Run:
OMP_NUM_THREADS=8 ./test_sole

Just for blocking:
gcc -O1 -mavx -mfma main_block.cpp sole_serial.cpp -o test_block
./test_block

Just for OpenMP:
gcc -O1 -fopenmp main_omp.cpp sole_omp.cpp -o test_openmp
OMP_NUM_THREADS=8 ./test_openmp
(Change above with however many threads you intend to use)

### Other Notes
To request a GPU on the SCC:

qrsh -l gpus=1 -l gpu_type=P100 -P ec527

This will request a 6.0 compute GPU.
Make sure to do "module load cuda" to get access to the nvcc compiler.