
#include <cstdio>
#include <cstdlib>
#include <math.h>

#define TILE_WIDTH        32
#define TOL            1e-2

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

// Part 1: MMM using global memory only.
__global__ void kernel_mmm_global (float* a, float* b, float* c, int length) {
  int Row = blockIdx.y * TILE_WIDTH + threadIdx.y; // row number
  int Col = blockIdx.x * TILE_WIDTH + threadIdx.x; // column number

  float tempSum = 0;

  if (Row < length && Col < length) { // compute only if not out of bounds
    // each iteration computes one value of C array (one row dotted with one col)
    for (int k = 0; k < length; ++k) { 
      tempSum += a[Row*length + k] * b[k*length+Col];
    }
    c[Row*length+Col] = tempSum;
  }
}

// Part 2: MMM using shared memory
__global__ void kernel_mmm_shared (float* a, float* b, float* c, int length) {
  __shared__ float a_s[TILE_WIDTH][TILE_WIDTH];
  __shared__ float b_s[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  float tempSum = 0;

  if (Row < length && Col < length) {
    for (int m = 0; m < length/TILE_WIDTH; ++m) {
      a_s[ty][tx] = a[Row*length + (m*TILE_WIDTH + tx)];
      b_s[ty][tx] = b[Col + (m*TILE_WIDTH + ty)*length];
      __syncthreads();

      for (int k = 0; k < TILE_WIDTH; ++k) {
        tempSum += a_s[ty][k] * b_s[k][tx];
      }
      __syncthreads();
    }
    c[Row*length+Col] = tempSum;
  }
}

// Part 3a: No memory coalescence, but no bank conflicts
__global__ void kernel_mmm_3a (float* a, float* b, float* c, int length) {
  __shared__ float a_s[TILE_WIDTH][TILE_WIDTH];
  __shared__ float b_s[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  // Transpose thread assignments to break memory coalescence
  int ty = threadIdx.x;
  int tx = threadIdx.y;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  float tempSum = 0;

  if (Row < length && Col < length) {
    for (int m = 0; m < length/TILE_WIDTH; ++m) {
      // Transpose a_s so that when it's being referenced,
      // There will not be memory bank conflicts
      a_s[tx][ty] = a[Row*length + (m*TILE_WIDTH + tx)];
      b_s[ty][tx] = b[Col + (m*TILE_WIDTH + ty)*length];
      __syncthreads();

      for (int k = 0; k < TILE_WIDTH; ++k) {
        tempSum += a_s[k][ty] * b_s[k][tx];
      }
      __syncthreads();
    }
    c[Row*length+Col] = tempSum;
  }
}

// Part 3b: Global memory coalescence, but has bank conflicts
__global__ void kernel_mmm_3b (float* a, float* b, float* c, int length) {
  __shared__ float a_s[TILE_WIDTH][TILE_WIDTH];
  __shared__ float b_s[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  float tempSum = 0;

  if (Row < length && Col < length) {
    for (int m = 0; m < length/TILE_WIDTH; ++m) {
      a_s[ty][tx] = a[Row*length + (m*TILE_WIDTH + tx)];
      // Transpose the b_s shared memory assignment
      // This forces a 32-way bank conflict
      // Since each thread in each warp accesses the same k'th bank
      b_s[tx][ty] = b[Col + (m*TILE_WIDTH + ty)*length];
      __syncthreads();

      for (int k = 0; k < TILE_WIDTH; ++k) {
        tempSum += a_s[ty][k] * b_s[tx][k];
      }
      __syncthreads();
    }
    c[Row*length+Col] = tempSum;
  }
}

// Part 3c: Mess with memory to make it bad!
__global__ void kernel_mmm_3c (float* a, float* b, float* c, int length) {
  __shared__ float a_s[TILE_WIDTH][TILE_WIDTH];
  __shared__ float b_s[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  // Transpose thread assignments to break memory coalescence
  int tx = threadIdx.y;
  int ty = threadIdx.x;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  float tempSum = 0;

  if (Row < length && Col < length) {
    for (int m = 0; m < length/TILE_WIDTH; ++m) {
      // Transpose the a_s shared memory assignment
      // This forces a 32-way bank conflict
      // Since each thread in each warp accesses the same k'th bank
      a_s[ty][tx] = a[Row*length + (m*TILE_WIDTH + tx)];
      b_s[ty][tx] = b[Col + (m*TILE_WIDTH + ty)*length];
      __syncthreads();

      for (int k = 0; k < TILE_WIDTH; ++k) {
        tempSum += a_s[ty][k] * b_s[k][tx];
      }
      __syncthreads();
    }
    c[Row*length+Col] = tempSum;
  }
}


void run_test(int arrLen, int grid_len, int block_len, char option) {
  // GPU Timing variables
  cudaEvent_t start, stop, start_mmm, stop_mmm;
  float elapsed_gpu;

  // Arrays on GPU global memory
  float *d_A, *d_B, *d_C;

  // Arrays on the host memory
  float *h_A, *h_B, *h_C, *h_C_dev;

  int i, errCount = 0, zeroCount = 0;
  size_t allocSize = arrLen * arrLen * sizeof(float);

  // printf("Length of the array = %d\n", arrLen);

  // Allocate GPU memory
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_A, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_B, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_C, allocSize));

  // Allocate arrays on host memory
  h_A                    = (float *) malloc(allocSize);
  h_B                    = (float *) malloc(allocSize);
  h_C                    = (float *) malloc(allocSize);
  h_C_dev                = (float *) malloc(allocSize);
  memset(h_C, 0, allocSize);

  // Initialize the host arrays
  // printf("\nInitializing the arrays ...");
  // Arrays are initialized with a known seed for reproducability
  initializeArray1D(h_A, arrLen, 456);
  initializeArray1D(h_B, arrLen, 123);
  // printf("\t... done\n\n");

  // Pre-compute array check
  // printf("Initial Array:\n");
  // printf("CPU:\n");
  // print_array(h_A, arrLen);
  // print_array(h_B, arrLen);
  // print_array(h_C, arrLen);
  // printf("GPU:\n");
  // print_array(h_A, arrLen);

  dim3 dimGrid(grid_len, grid_len);
  dim3 dimBlock(block_len, block_len);
  printf("Array Size: %dx%d | Block Size: %dx%d | Grid Size: %dx%d\n", 
          arrLen, arrLen, block_len, block_len, grid_len, grid_len);


  // Timer for End-to-End Operation
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // Record event on the default stream
  cudaEventRecord(start, 0);

  // Transfer the arrays to the GPU memory
  CUDA_SAFE_CALL(cudaMemcpy(d_A, h_A, allocSize, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_B, h_B, allocSize, cudaMemcpyHostToDevice));

  // Timer only for MMM Computation
  cudaEventCreate(&start_mmm);
  cudaEventCreate(&stop_mmm);
  // Record event on the default stream
  cudaEventRecord(start_mmm, 0);

  // Launch the kernel
  if (option == '1') {
    kernel_mmm_global<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, arrLen);
  }
  else if (option == '2') {
    kernel_mmm_shared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, arrLen);
  }
  else if (option == 'a') {
    kernel_mmm_3a<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, arrLen);
  }
  else if (option == 'b') {
    kernel_mmm_3b<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, arrLen);
  }
  else if (option == 'c') {
    kernel_mmm_3c<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, arrLen);
  }

  // Stop and destroy the timer
  cudaEventRecord(stop_mmm,0);
  cudaEventSynchronize(stop_mmm);
  cudaEventElapsedTime(&elapsed_gpu, start_mmm, stop_mmm);
  printf("GPU MMM Only time: %f (msec)\n", elapsed_gpu);
  cudaEventDestroy(start_mmm);
  cudaEventDestroy(stop_mmm);

  // Check for errors during launch
  CUDA_SAFE_CALL(cudaPeekAtLastError());

  // Transfer the results back to the host
  CUDA_SAFE_CALL(cudaMemcpy(h_C_dev, d_C, allocSize, cudaMemcpyDeviceToHost));


  // Stop and destroy the timer
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_gpu, start, stop);
  printf("GPU End-to-End time: %f (msec)\n", elapsed_gpu);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);


  printf("CPU Time: ");
  // Create the cuda events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // Record event on the default stream
  cudaEventRecord(start, 0);

  // Compute the results on the host
  mmm(h_A, h_B, h_C, arrLen, 8);

  // Stop and destroy the timer
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_gpu, start, stop);
  printf("%f (msec)\n", elapsed_gpu);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Results:
  // printf("Initial Array:\n");
  // printf("CPU:\n");
  // print_array(h_A, arrLen);
  // print_array(h_B, arrLen);
  // print_array(h_C, arrLen);

  // Compare the results
  float errTotal = 0;
  for(i = 0; i < arrLen*arrLen; i++) {
    if (abs(h_C_dev[i] - h_C[i]) > TOL) {
      errCount++;
      errTotal += abs(h_C_dev[i] - h_C[i]);
    }
    if (h_C_dev[i] == 0) {
      zeroCount++;
    }
  }

  // printf("\nGPU Results:\n");
  // print_array(h_C_dev, arrLen);

  if (errCount > 0) {
    printf("\n@ERROR: TEST FAILED: %d results did not match\n", errCount);
    printf("Total error is %.3f\n", errTotal);
  }
  else if (zeroCount > 0){
    printf("\n@ERROR: TEST FAILED: %d results (from GPU) are zero\n", zeroCount);
  }
  else {
    printf("TEST PASSED: All results matched\n");
  }

  // Free-up device and host memory
  CUDA_SAFE_CALL(cudaFree(d_A));
  CUDA_SAFE_CALL(cudaFree(d_B));
  CUDA_SAFE_CALL(cudaFree(d_C));

  free(h_A); free(h_B); free(h_C); free(h_C_dev);
}


// int main(int argc, char **argv){
//   // Select GPU
//   CUDA_SAFE_CALL(cudaSetDevice(0));
  
//   printf("\nRunning Part 1\n");
//   run_test(1024, 1024/TILE_WIDTH, TILE_WIDTH, '1');
//   run_test(2048, 2048/TILE_WIDTH, TILE_WIDTH, '1');

//   printf("\nRunning Part 2\n");
//   run_test(1024, 1024/TILE_WIDTH, TILE_WIDTH, '2');
//   run_test(2048, 2048/TILE_WIDTH, TILE_WIDTH, '2');

//   printf("\nRunning Part 3a\n");
//   run_test(1024, 1024/TILE_WIDTH, TILE_WIDTH, 'a');
//   run_test(2048, 2048/TILE_WIDTH, TILE_WIDTH, 'a');

//   printf("\nRunning Part 3b\n");
//   run_test(1024, 1024/TILE_WIDTH, TILE_WIDTH, 'b');
//   run_test(2048, 2048/TILE_WIDTH, TILE_WIDTH, 'b');

//   printf("\nRunning Part 3c\n");
//   run_test(1024, 1024/TILE_WIDTH, TILE_WIDTH, 'c');
//   run_test(2048, 2048/TILE_WIDTH, TILE_WIDTH, 'c');
//   return 0;
// }