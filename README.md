# Dense Systems of Linear Equations
### EC527 Final Project, Spring 2026
### Jiaxing Wang, Alvin Yan, Owen Jiang
---
### Summary
This project explores the implementation and optimization of dense systems of linear equations. We start with a naive implementation of solving $$Ax=b$$ through LU decomposition and forward and backward passes. Then, we optimize LU decomposition through blocking, vectorization, and parallelization through OpenMP and CUDA.

### Implementations
#### Serial
1. Serial Naive (*sole_serial*)
2. Serial Blocked (*sole_blocked*)
#### Vectorized
3. AVX (*sole_avx*)
#### OpenMP
4. OpenMP Naive (*sole_omp*)
5. OpenMP Alternate Access Pattern (*sole_omp_altaccess*)
6. OpenMP Attempts at Optimization (not good) (*sole_omp_optimized*)
#### CUDA
7. CUDA Single-Kernel (*sole_cuda_combine*)
8. CUDA Two Kernels, Global Memory (*sole_cuda*)
9. CUDA Two Kernels, Local Memory (*sole_cuda_local*)
#### The Benchmark
10. cuBLAS (Using cuBLAS library) (*cuBLAS*)

To implement in the future: OpenMP and CUDA with Shared Memory and Blocking

### Compilation Notes
#### To test all implementations, compile:
    nvcc -O1 -Xcompiler "-fopenmp,-mavx,-mfma" main.cu sole_serial.cpp sole_omp.cpp sole_gpu.cu -o test_sole -lcusolver -lcublas -lm 

**To execute:** OMP_NUM_THREADS=8 ./test_sole

#### To test blocking only, compile:
    gcc -O1 -mavx -mfma main_block.cpp sole_serial.cpp -o test_block
**To execute:** ./test_block

#### To test OpenMP only, compile:
    gcc -O1 -fopenmp main_omp.cpp sole_omp.cpp -o test_openmp
**To execute:** OMP_NUM_THREADS=8 ./test_openmp

(Change above with however many threads you intend to use)

#### To test GPU block sizes only, compile:
    nvcc -O1 main_gpu.cu sole_gpu.cu -o test_gpu -lcusolver -lcublas -lm
**To execute:** ./test_gpu

### Other Notes
To request a GPU on the SCC: *qrsh -l gpus=1 -l gpu_type=P100 -P ec527*

This will request a 6.0 compute GPU. Make sure to run **module load cuda** in the terminal to get access to the nvcc compiler.