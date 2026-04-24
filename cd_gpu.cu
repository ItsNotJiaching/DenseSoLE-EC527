/*
This code explores attempting to perform memory optimizations like coalesced and 
*/

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <omp.h>
#include "sole.cuh"

#define IMUL(a, b) __mul24(a, b)

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                                       cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

void print_mat(data_t* mat, int row_len) {
  for (int i=0; i < row_len; i++) {
    for (int j=0; j < row_len; j++) {
      printf("%.3f ", mat[i*row_len + j]);
    }
    printf("\n");
  }
  printf("\n");
}

__global__ void kernel_lu_step(data_t* A, int k, int row_len) {
    int col = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int row = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;

    if (row < row_len && col < row_len) {
        A[row * row_len + col] -= A[row * row_len + k] * A[k * row_len + col];
    }
}

/**
 * Takes the input data (each size being tested) 
 */
void sole_cuda(data_t* A, data_t* x, data_t* b, int row_len, int grid_len, int block_len) {
  // Select GPU
  CUDA_SAFE_CALL(cudaSetDevice(0));

  // Arrays on GPU global memory
  data_t *A_GPU;
  size_t matSize = row_len * row_len * sizeof(data_t);
  // Allocate GPU memory
  CUDA_SAFE_CALL(cudaMalloc((void **)&A_GPU, matSize));

  // Transfer the arrays to the GPU memory
  CUDA_SAFE_CALL(cudaMemcpy(A_GPU, A, matSize, cudaMemcpyHostToDevice));

  // Pre-compute array check
  // printf("Initial Host Array:\n");
  // print_mat(A, row_len);
  // print_mat(x_GPU, row_len);
  // print_mat(b_GPU, row_len);

  // Grid and Block Dimensions, Terminal Output
  dim3 dimGrid(grid_len, grid_len);
  dim3 dimBlock(block_len, block_len);
  // printf("Array Size: %dx%d | Block Size: %dx%d | Grid Size: %dx%d\n", 
  //         row_len, row_len, block_len, block_len, grid_len, grid_len);
  
  // Launch the kernel
  // One kernel per pivot step — CUDA serializes launches on same stream
  for (int k = 0; k < row_len - 1; k++) {
    int rows_remaining = row_len - k - 1;    // threads needed = rows below pivot
    int blockSize = 1024;
    int gridSize = (rows_remaining + blockSize - 1) / blockSize;

    kernel_lu_step<<<gridSize, blockSize>>>(A_GPU, k, row_len);

    CUDA_SAFE_CALL(cudaPeekAtLastError());
  }

  // Transfer the results back to the host
  CUDA_SAFE_CALL(cudaMemcpy(A, A_GPU, matSize, cudaMemcpyDeviceToHost));

  // print_mat(A, row_len);

  // forward sub Ly = b (uses x instead of y for better spatial locality)
  for (int i = 0; i < row_len; i++) {
      data_t* row = &A[i * row_len];
      data_t sum = 0.0; //intermediatary sum for dot product
      #pragma omp parallel for reduction(-:sum)
      for (int j = 0; j < i; j++) //this basically creates L staircase
          sum += row[j] * x[j]; //lower half, basically. y[i] = b[i] - A[i*row_len] * y[j]. 
      x[i] = b[i] - sum;
      // x[i] /= 1.0 //divide by diagonal per formula. For lower diagonal, it's always one, so no point of calculating.
  }

  //back sub Ux = y
  for (int i = row_len - 1; i >= 0; i--) {
      data_t* row = &A[i * row_len];
      data_t sum = 0.0; //intermediatary sum for dot product
      #pragma omp parallel for reduction(-:sum)
      for (int j = i + 1; j < row_len; j++) //get ahead of diagonal to iterate through U
          sum += row[j] * x[j]; //upper half, basically. x[i] = y[i] - A[i*row_len] * x[j]. 
      x[i] = x[i] - sum; //writing existing y[i] into actual x[i]
      x[i] = x[i]/row[i]; //divide by diagonal per the formula 
  }

  // Free-up device and host memory
  CUDA_SAFE_CALL(cudaFree(A_GPU));
}