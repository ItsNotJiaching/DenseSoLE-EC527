// Select data type of calculations here
typedef double data_t;

// Helper functions
void detect_threads_setting();
double interval(struct timespec start, struct timespec end);
void print_array(data_t* v, int arr_len);
void init_matrix(data_t *mat, int len);
void init_vector(data_t *mat, int len);
double verify(data_t* arrA, data_t* arrX, data_t* arrB, int n);

// Serial versions
void sole_serial(data_t* A, data_t* x, data_t* b, int row_len);
void sole_blocked(data_t* A, data_t* x, data_t* b, int row_len, int B);

// AVX versions
void sole_avx(data_t* A, data_t* x, data_t* b, int row_len);

// OpenMP versions
void sole_omp_naive(data_t* A, data_t* x, data_t* b, int row_len);
void sole_omp_altload(data_t* A, data_t* x, data_t* b, int row_len);
void sole_omp_optimized(data_t* A, data_t* x, data_t* b, int row_len);
void sole_omp_blocked(data_t* A, data_t* x, data_t* b, int row_len, int B);
void sole_omp_tiled_unrolling(data_t* A, data_t* x, data_t* b, int row_len, int B, int T);

// CUDA versions
void sole_cuda(data_t* A, data_t* x, data_t* b, int row_len, int block_len);
void sole_cuda_local(data_t* A, data_t* x, data_t* b, int row_len, int block_len);
void sole_cuda_combine(data_t* A, data_t* x, data_t* b, int row_len, int block_len);

// Benchmark function
void cuBLAS(data_t* A, data_t* x, data_t* b, int row_len);
