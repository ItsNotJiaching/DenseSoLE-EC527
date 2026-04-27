typedef double data_t;

void detect_threads_setting();
void sole_cuda(data_t* A, data_t* x, data_t* b, int row_len, int block_len);
void sole_omp_naive(data_t* A, data_t* x, data_t* b, int row_len);
void sole_omp_altload(data_t* A, data_t* x, data_t* b, int row_len);
void sole_omp_optimized(data_t* A, data_t* x, data_t* b, int row_len);
void sole_serial(data_t* A, data_t* x, data_t* b, int row_len);
void sole_blocked1(data_t* A, data_t* x, data_t* b, int row_len, int B);
void sole_blocked2(data_t* A, data_t* x, data_t* b, int row_len, int B);
void sole_avx(data_t* A, data_t* x, data_t* b, int row_len);
double interval(struct timespec start, struct timespec end);
void cuBLAS(data_t* A, data_t* x, data_t* b, int row_len, int grid_len, int block_len);