typedef float data_t;

void detect_threads_setting();
void test_cuda(data_t* A, data_t* x, data_t* b, int row_len);
void sole_omp_naive(data_t* A, data_t* x, data_t* b, int row_len);
void sole_omp_balanced(data_t* A, data_t* x, data_t* b, int row_len);
void sole_serial(data_t* A, data_t* x, data_t* b, int row_len);
void sole_blocked(data_t* A, data_t* x, data_t* b, int row_len, int B);
