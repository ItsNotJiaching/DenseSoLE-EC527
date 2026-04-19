typedef float data_t;

void test_cuda(data_t* A, data_t* x, data_t* b, int row_len);
void sole_omp(data_t* A, data_t* x, data_t* b, int row_len);
void sole_serial(data_t* A, data_t* x, data_t* b, int row_len);
void sole_blocked(data_t* A, data_t* x, data_t* b, int row_len, int B);