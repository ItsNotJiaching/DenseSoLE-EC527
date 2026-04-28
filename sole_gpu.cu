  #include <cstdio>
  #include <cstdlib>
  #include <math.h>
  #include <omp.h>
  #include "sole.cuh"
  #include <cusolverDn.h> 

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

/**
 * Lower step of LU Decomposition using global memory.
 * 
 * @param A pointer to input matrix, the A in the Ax=b.
 * @param k k'th pivot iteration of LU decomposition
 * @param row_len Row length of array (total size would be row_len^2)
 * @return No return; computed outputs are stored in x.
 * @author Jiaxing Wang
 */
__global__ void lower_step(data_t* A, int k, int row_len) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int row = k + 1 + tid;
  if (row >= row_len) return;
  
  // Compute Lower Triangular
  A[row*row_len + k] = A[row*row_len + k] /A[k*row_len + k];
}

/**
 * Upper step of LU Decomposition using global memory.
 * 
 * @param A pointer to input matrix, the A in the Ax=b.
 * @param k k'th pivot iteration of LU decomposition
 * @param row_len Row length of array (total size would be row_len^2)
 * @return No return; computed outputs are stored in x.
 * @author Jiaxing Wang
 */
__global__ void upper_step(data_t* A, int k, int row_len) {
  int row = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
  int col = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_len || col >= row_len) return;
  // Compute Remaining Elements in Rows right of and under pivot
  A[row*row_len + col] -= A[row*row_len + k] * A[k*row_len + col];
}

/**
 * Lower step of LU Decomposition using local memory.
 * 
 * @param A pointer to input matrix, the A in the Ax=b.
 * @param k k'th pivot iteration of LU decomposition
 * @param row_len Row length of array (total size would be row_len^2)
 * @return No return; computed outputs are stored in x.
 * @author Jiaxing Wang
 */
__global__ void lower_step_local(data_t* A, int k, int row_len) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int row = k + 1 + tid;
  if (row >= row_len) return;

  // Declare shared memory
  data_t elem = A[row*row_len + k];
  data_t pivot = A[k*row_len + k];

  // compute lower triangular (L)
  elem = elem / pivot;
  A[row*row_len+k] = elem;
}

/**
 * Upper step of LU Decomposition using local memory.
 * 
 * @param A pointer to input matrix, the A in the Ax=b.
 * @param k k'th pivot iteration of LU decomposition
 * @param row_len Row length of array (total size would be row_len^2)
 * @return No return; computed outputs are stored in x.
 * @author Jiaxing Wang
 */
__global__ void upper_step_local(data_t* A, int k, int row_len) {
  int row = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
  int col = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_len || col >= row_len) return;

  // Declare local memory
  data_t elem = A[row*row_len + col];
  data_t multiplier = A[row*row_len + k];
  __shared__ data_t elem_pivot;
  elem_pivot = A[k*row_len + col];

  // compute matrix update
  elem -= multiplier * elem_pivot;
  A[row*row_len + col] = elem;
}

/**
 * Single-kernel version of LU Decomposition where each row is assigned a thread,
 * so row updates are done sequentially.
 * 
 * @param A pointer to input matrix, the A in the Ax=b.
 * @param k k'th pivot iteration of LU decomposition
 * @param row_len Row length of array (total size would be row_len^2)
 * @return No return; computed outputs are stored in x.
 * @author Jiaxing Wang
 */
__global__ void lower_upper_step(data_t* A, int k, int row_len) {
  extern __shared__ data_t pivot_row[];  // row_len - k - 1 elements

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int row = k + 1 + tid;
  int cols = row_len - k - 1;

  // All threads in block collaboratively load pivot row from global -> shared
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
      pivot_row[j] = A[k*row_len + k+1 + j];
  }
  __syncthreads();

  if (row >= row_len) return;

  // Lower
  A[row*row_len + k] /= A[k*row_len + k];
  data_t multiplier = A[row*row_len + k];

  // Upper: reads from shared instead of global
  for (int j = 0; j < cols; j++) {
      A[row*row_len + k+1 + j] -= multiplier * pivot_row[j];
  }
}

/**
 * CUDA implementation of SoLE done with single CUDA kernel.
 * @param A pointer to input matrix, the A in the Ax=b.
 * @param x pointer to output vector, the x in the Ax=b.
 * @param b pointer to the b in the Ax=b.
 * @param row_len Row length of array (total size would be row_len^2)
 * @param blockSize Size of CUDA block (number of threads per block)
 * @author Jiaxing Wang
 */
void sole_cuda_combine(data_t* A, data_t* x, data_t* b, int row_len, int blockSize) {
  // Select GPU
  CUDA_SAFE_CALL(cudaSetDevice(0));

  // Arrays on GPU global memory
  data_t *A_GPU;
  size_t matSize = row_len * row_len * sizeof(data_t);
  // Allocate GPU memory, then transfer array to GPU
  CUDA_SAFE_CALL(cudaMalloc((void **)&A_GPU, matSize));
  CUDA_SAFE_CALL(cudaMemcpy(A_GPU, A, matSize, cudaMemcpyHostToDevice));
  
  // // GPU Timing variables
  // cudaEvent_t start, stop;
  // float elapsed_gpu;
  // // Timer for End-to-End Operation
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // // Record event on the default stream
  // cudaEventRecord(start, 0);

  // Launch the kernel
  // One kernel per pivot step — CUDA serializes launches on same stream
  for (int k = 0; k < row_len - 1; k++) {
    int rows_remaining = row_len - k - 1;
    int gridSize = (rows_remaining + blockSize - 1) / blockSize;
    size_t sharedMem = (row_len - k - 1) * sizeof(data_t);

    lower_upper_step<<<gridSize, blockSize, sharedMem>>>(A_GPU, k, row_len);
  }

  // // Stop and destroy the timer
  // cudaEventRecord(stop,0);
  // cudaEventSynchronize(stop);
  // cudaEventElapsedTime(&elapsed_gpu, start, stop);
  // printf("GPU LU Decomposition Only time: %.2f ms\n", elapsed_gpu);
  // cudaEventDestroy(start);
  // cudaEventDestroy(stop);

  // Transfer the results back to the host
  CUDA_SAFE_CALL(cudaMemcpy(A, A_GPU, matSize, cudaMemcpyDeviceToHost));

  // forward sub Ly = b (uses x instead of y for better spatial locality)
  for (int i = 0; i < row_len; i++) {
      data_t* row = &A[i * row_len];
      data_t sum = 0.0; //intermediary sum for dot product
      #pragma omp parallel for reduction(-:sum)
      for (int j = 0; j < i; j++) //this basically creates L staircase
          sum += row[j] * x[j]; //lower half, basically. y[i] = b[i] - A[i*row_len] * y[j]. 
      x[i] = b[i] - sum;
      // x[i] /= 1.0 //divide by diagonal per formula. For lower diagonal, it's always one, so no point of calculating.
  }

  // back sub Ux = y
  for (int i = row_len - 1; i >= 0; i--) {
      data_t* row = &A[i * row_len];
      data_t sum = 0.0; //intermediatary sum for dot product
      #pragma omp parallel for reduction(-:sum)
      for (int j = i + 1; j < row_len; j++) //get ahead of diagonal to iterate through U
          sum += row[j] * x[j]; //upper half, basically. x[i] = y[i] - A[i*row_len] * x[j]. 
      x[i] = x[i] - sum; //writing existing y[i] into actual x[i]
      x[i] = x[i]/row[i]; //divide by diagonal per the formula 
  }

  // Free up GPU memory
  CUDA_SAFE_CALL(cudaFree(A_GPU));
}

/**
 * CUDA implementation of SoLE done with two CUDA kernels using local memory.
 * @param A pointer to input matrix, the A in the Ax=b.
 * @param x pointer to output vector, the x in the Ax=b.
 * @param b pointer to the b in the Ax=b.
 * @param row_len Row length of array (total size would be row_len^2)
 * @param blockSize Size of CUDA block (number of threads per block)
 * @author Jiaxing Wang
 */
void sole_cuda_local(data_t* A, data_t* x, data_t* b, int row_len, int blockSize) {
  // Select GPU
  CUDA_SAFE_CALL(cudaSetDevice(0));

  // Arrays on GPU global memory
  data_t *A_GPU;
  size_t matSize = row_len * row_len * sizeof(data_t);
  // Allocate GPU memory, then transfer array to GPU
  CUDA_SAFE_CALL(cudaMalloc((void **)&A_GPU, matSize));
  CUDA_SAFE_CALL(cudaMemcpy(A_GPU, A, matSize, cudaMemcpyHostToDevice));
  
  // // GPU Timing variables
  // cudaEvent_t start, stop;
  // float elapsed_gpu;
  // // Timer for End-to-End Operation
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // // Record event on the default stream
  // cudaEventRecord(start, 0);

  // Launch the kernel
  // One kernel per pivot step — CUDA serializes launches on same stream
  for (int k = 0; k < row_len - 1; k++) {
    int rows_remaining = row_len - k - 1;    // threads needed = rows below pivot
    int gridSize = (rows_remaining + blockSize - 1) / blockSize;

    lower_step_local<<<gridSize, blockSize>>>(A_GPU, k, row_len);

    dim3 blockSizeRemain(16, 16);
    dim3 gridSizeRemain(
        (rows_remaining + 15) / 16,
        (rows_remaining + 15) / 16
    );
    upper_step_local<<<gridSizeRemain, blockSizeRemain>>>(A_GPU, k, row_len);

    CUDA_SAFE_CALL(cudaPeekAtLastError());
  }

  // // Stop and destroy the timer
  // cudaEventRecord(stop,0);
  // cudaEventSynchronize(stop);
  // cudaEventElapsedTime(&elapsed_gpu, start, stop);
  // printf("GPU LU Decomposition Only time: %.2f ms\n", elapsed_gpu);
  // cudaEventDestroy(start);
  // cudaEventDestroy(stop);

  // Transfer the results back to the host
  CUDA_SAFE_CALL(cudaMemcpy(A, A_GPU, matSize, cudaMemcpyDeviceToHost));

  // forward sub Ly = b (uses x instead of y for better spatial locality)
  for (int i = 0; i < row_len; i++) {
      data_t* row = &A[i * row_len];
      data_t sum = 0.0; //intermediary sum for dot product
      #pragma omp parallel for reduction(-:sum)
      for (int j = 0; j < i; j++) //this basically creates L staircase
          sum += row[j] * x[j]; //lower half, basically. y[i] = b[i] - A[i*row_len] * y[j]. 
      x[i] = b[i] - sum;
      // x[i] /= 1.0 //divide by diagonal per formula. For lower diagonal, it's always one, so no point of calculating.
  }

  // back sub Ux = y
  for (int i = row_len - 1; i >= 0; i--) {
      data_t* row = &A[i * row_len];
      data_t sum = 0.0; //intermediatary sum for dot product
      #pragma omp parallel for reduction(-:sum)
      for (int j = i + 1; j < row_len; j++) //get ahead of diagonal to iterate through U
          sum += row[j] * x[j]; //upper half, basically. x[i] = y[i] - A[i*row_len] * x[j]. 
      x[i] = x[i] - sum; //writing existing y[i] into actual x[i]
      x[i] = x[i]/row[i]; //divide by diagonal per the formula 
  }

  // Free up GPU memory
  CUDA_SAFE_CALL(cudaFree(A_GPU));
}

/**
 * CUDA implementation of SoLE done with two CUDA kernels, using global memory.
 * @param A pointer to input matrix, the A in the Ax=b.
 * @param x pointer to output vector, the x in the Ax=b.
 * @param b pointer to the b in the Ax=b.
 * @param row_len Row length of array (total size would be row_len^2)
 * @param blockSize Size of CUDA block (number of threads per block)
 * @author Jiaxing Wang
 */
void sole_cuda(data_t* A, data_t* x, data_t* b, int row_len, int blockSize) {
  // Select GPU
  CUDA_SAFE_CALL(cudaSetDevice(0));

  // Arrays on GPU global memory
  data_t *A_GPU;
  size_t matSize = row_len * row_len * sizeof(data_t);
  // Allocate GPU memory, then transfer array to GPU
  CUDA_SAFE_CALL(cudaMalloc((void **)&A_GPU, matSize));
  CUDA_SAFE_CALL(cudaMemcpy(A_GPU, A, matSize, cudaMemcpyHostToDevice));

  // // GPU Timing variables
  // cudaEvent_t start, stop;
  // float elapsed_gpu;
  // // Timer for End-to-End Operation
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // // Record event on the default stream
  // cudaEventRecord(start, 0);

  // Launch the kernel
  // One kernel per pivot step — CUDA serializes launches on same stream
  for (int k = 0; k < row_len - 1; k++) {
    int rows_remaining = row_len - k - 1;    // threads needed = rows below pivot
    int gridSize = (rows_remaining + blockSize - 1) / blockSize;

    lower_step<<<gridSize, blockSize>>>(A_GPU, k, row_len);

    dim3 blockSizeRemain(16, 16);
    dim3 gridSizeRemain(
        (rows_remaining + 15) / 16,
        (rows_remaining + 15) / 16
    );
    upper_step<<<gridSizeRemain, blockSizeRemain>>>(A_GPU, k, row_len);

    CUDA_SAFE_CALL(cudaPeekAtLastError());
  }

  // // Stop and destroy the timer
  // cudaEventRecord(stop,0);
  // cudaEventSynchronize(stop);
  // cudaEventElapsedTime(&elapsed_gpu, start, stop);
  // printf("GPU LU Decomposition Only time: %.2f ms\n", elapsed_gpu);
  // cudaEventDestroy(start);
  // cudaEventDestroy(stop);

  // Transfer the results back to the host
  CUDA_SAFE_CALL(cudaMemcpy(A, A_GPU, matSize, cudaMemcpyDeviceToHost));

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

  // Free up GPU memory
  CUDA_SAFE_CALL(cudaFree(A_GPU));
}

/**
 * cuBLAS (NVIDIA's library) implementation of SoLE.
 * @param A pointer to input matrix, the A in the Ax=b.
 * @param x pointer to output vector, the x in the Ax=b.
 * @param b pointer to the b in the Ax=b.
 * @param row_len Row length of array (total size would be row_len^2)
 * @param blockSize Size of CUDA block (number of threads per block)
 * @author Owen Jiang 
 */
void cuBLAS(data_t* A, data_t* x, data_t* b, int row_len) {
  // Select GPU
  CUDA_SAFE_CALL(cudaSetDevice(0));

  // Arrays on GPU global memory
  data_t *A_GPU, *b_GPU;
  int *ipiv_GPU, *info_GPU;
  size_t matSize = row_len * row_len * sizeof(data_t);
  size_t vecSize = row_len * sizeof(data_t);

  // Allocate GPU memory
  CUDA_SAFE_CALL(cudaMalloc((void **)&A_GPU,    matSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&b_GPU,    vecSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&ipiv_GPU, row_len * sizeof(int)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&info_GPU, sizeof(int)));

  // Transfer arrays to GPU
  CUDA_SAFE_CALL(cudaMemcpy(A_GPU, A, matSize, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(b_GPU, b, vecSize, cudaMemcpyHostToDevice));

  // Create cuSOLVER handle
  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  // Query workspace size (needed for cuBLAS)
  int lwork;
  cusolverDnDgetrf_bufferSize(handle, row_len, row_len, A_GPU, row_len, &lwork); //change to cusolverDnDgetrf_bufferSize for float
  data_t* workspace;
  CUDA_SAFE_CALL(cudaMalloc((void **)&workspace, lwork * sizeof(data_t)));

  // // GPU Timing variables
  // cudaEvent_t start, stop;
  // float elapsed_gpu;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start, 0);

  // LU Factorization 
  cusolverDnDgetrf(handle, row_len, row_len, A_GPU, row_len, workspace, NULL, info_GPU); //change to cusolverDnSgetrf for float

  // Triangular solve — writes solution into b_GPU
  cusolverDnDgetrs(handle, CUBLAS_OP_T, row_len, 1, A_GPU, row_len, NULL, b_GPU, row_len, info_GPU); //change to cusolverDnSgetrs for float

  // // Stop timer
  // cudaEventRecord(stop, 0);
  // cudaEventSynchronize(stop);
  // cudaEventElapsedTime(&elapsed_gpu, start, stop);
  // printf("cuSOLVER LU + Solve time: %.2f ms\n", elapsed_gpu);
  // cudaEventDestroy(start);
  // cudaEventDestroy(stop);

  // Check for errors
  int info_host;
  CUDA_SAFE_CALL(cudaMemcpy(&info_host, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
  if (info_host != 0)
      printf("  cuSOLVER WARNING: info = %d (singular matrix?)\n", info_host);

  // Transfer results back — solution is in b_GPU
  CUDA_SAFE_CALL(cudaMemcpy(x, b_GPU, vecSize, cudaMemcpyDeviceToHost));

  // Cleanup
  cusolverDnDestroy(handle);
  CUDA_SAFE_CALL(cudaFree(A_GPU));
  CUDA_SAFE_CALL(cudaFree(b_GPU));
  CUDA_SAFE_CALL(cudaFree(ipiv_GPU));
  CUDA_SAFE_CALL(cudaFree(info_GPU));
  CUDA_SAFE_CALL(cudaFree(workspace));
}
