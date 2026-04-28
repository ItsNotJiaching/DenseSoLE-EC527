# Dense Systems of Linear Equations
### EC527 Final Project, Spring 2026
### Jiaxing Wang, Alvin Yan, Owen Jiang
---
### Summary
This project explores the implementation and optimization of dense systems of linear equations. We start with a naive implementation of solving $$Ax=b$$ through LU decomposition and forward and backward passes. Then, we optimize LU decomposition through blocking, vectorization, and parallelization through OpenMP and CUDA.

### Implementations
#### Serial
1. Serial Naive
2. Serial Blocked
#### Vectorized
3. AVX
#### OpenMP
4. OpenMP Naive
5. OpenMP Alternate Access Pattern
6. OpenMP Attempts at Optimization (not good)
#### CUDA
7. CUDA Single-Kernel
8. CUDA Two Kernels, Global Memory
9. CUDA Two Kernels, Local Memory
#### The Benchmark
10. cuBLAS (Using cuBLAS library)

To implement in the future: OpenMP and CUDA with Shared Memory and Blocking

### Compilation Notes
#### Test all implementations:
    nvcc -O1 -Xcompiler "-fopenmp,-mavx,-mfma" main.cu sole_serial.cpp sole_omp.cpp sole_gpu.cu -o test_sole -lcusolver -lcublas -lm 

**To run:** OMP_NUM_THREADS=8 ./test_sole

#### Test blocking only:
    gcc -O1 -mavx -mfma main_block.cpp sole_serial.cpp -o test_block
**To run:** ./test_block

#### Test OpenMP only:
    gcc -O1 -fopenmp main_omp.cpp sole_omp.cpp -o test_openmp
**To run:** OMP_NUM_THREADS=8 ./test_openmp
(Change above with however many threads you intend to use)

### Other Notes
To request a GPU on the SCC: *qrsh -l gpus=1 -l gpu_type=P100 -P ec527*
This will request a 6.0 compute GPU.
Make sure to run **module load cuda** in the terminal to get access to the nvcc compiler.