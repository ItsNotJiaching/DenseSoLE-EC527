  #include <cstdio>
  #include <cstdlib>
  #include <math.h>
  #include <omp.h>
  #include "sole.cuh"
  #include <cusolverDn.h> 

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

__global__ void kernel_upper_step(data_t* A, int k, int row_len) {
  int row = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
  int col = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_len || col >= row_len) return;
  // Compute Remaining Elements in Rows right of and under pivot
  A[row*row_len + col] -= A[row*row_len + k] * A[k*row_len + col];
}

  __global__ void kernel_lu_step(data_t* A, int k, int row_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int row = k + 1 + tid;
  if (row >= row_len) return;
  
  // Compute Lower Triangular
  A[row*row_len + k] = A[row*row_len + k] /A[k*row_len + k];
}
/* NAIVE CUDA-PARALLEL, THEN SERIAL FOR THE GRID RIGHT OF + BELOW PIVOT UPDATES */
// __global__ void kernel_lu_step(data_t* A, int k, int row_len) {
//   int tid = blockIdx.x * blockDim.x + threadIdx.x;

//   int row = k + 1 + tid;
//   if (row >= row_len) return;
  
//   // Compute Lower Triangular
//   A[row*row_len + k] = A[row*row_len + k] /A[k*row_len + k];

//   // Compute Upper Triangular
//   for (int j = k+1; j < row_len; j++) {
//     A[row*row_len + j] -= A[row*row_len + k] * A[k*row_len + j];
//   }
// }

  /**
   * Takes the input data (each size being tested) 
   */
void sole_cuda(data_t* A, data_t* x, data_t* b, int row_len, int blockSize) {
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
  
  // GPU Timing variables
  cudaEvent_t start, stop;
  float elapsed_gpu;

  // Timer for End-to-End Operation
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // Record event on the default stream
  cudaEventRecord(start, 0);

  // Launch the kernel
  // One kernel per pivot step — CUDA serializes launches on same stream
  for (int k = 0; k < row_len - 1; k++) {
    int rows_remaining = row_len - k - 1;    // threads needed = rows below pivot
    int gridSize = (rows_remaining + blockSize - 1) / blockSize;

    kernel_lu_step<<<gridSize, blockSize>>>(A_GPU, k, row_len);

    dim3 blockSizeRemain(16, 16);
    dim3 gridSizeRemain(
        (rows_remaining + 15) / 16,
        (rows_remaining + 15) / 16
    );
    kernel_upper_step<<<gridSizeRemain, blockSizeRemain>>>(A_GPU, k, row_len);

    CUDA_SAFE_CALL(cudaPeekAtLastError());
  }

  // Stop and destroy the timer
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_gpu, start, stop);
  printf("GPU LU Decomposition Only time: %.2f ms\n", elapsed_gpu);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

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

  void cuBLAS(data_t* A, data_t* x, data_t* b, int row_len, int grid_len, int block_len) {
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

    // GPU Timing variables
    cudaEvent_t start, stop;
    float elapsed_gpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // LU Factorization 
    cusolverDnDgetrf(handle, row_len, row_len, A_GPU, row_len, workspace, NULL, info_GPU); //change to cusolverDnSgetrf for float

    // Triangular solve — writes solution into b_GPU
    cusolverDnDgetrs(handle, CUBLAS_OP_T, row_len, 1, A_GPU, row_len, NULL, b_GPU, row_len, info_GPU); //change to cusolverDnSgetrs for float

    // Stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_gpu, start, stop);
    printf("cuSOLVER LU + Solve time: %.2f ms\n", elapsed_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

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
