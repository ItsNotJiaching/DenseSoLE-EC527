
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include "sole_multi.h"
#include "sole_serial.h"
#include "sole_gpu.cuh"

void initializeArray1D(float *arr, int len, int seed);
void print_array(float* v, int arr_len);


// Let's do multiples of 8 for now
#define A   8  /* coefficient of x^2 */
#define B   8  /* coefficient of x */
#define C   16  /* constant term */

#define NUM_TESTS 15   /* Number of different sizes to test */

#define OPTIONS 3


int main(){
  int OPTION;
  struct timespec time_start, time_stop;
  double time_stamp[OPTIONS][NUM_TESTS];
  double wakeup_answer;
  long int x, n, alloc_size;
  wakeup_answer = wakeup_delay();

  x = NUM_TESTS-1;
  alloc_size = A*x*x + B*x + C;

  // Terminal Output
  printf("Running Dense System of Linear Equations (SoLE) Test ");
  printf("for %d different matrix sizes from %d to %d\n\n", NUM_TESTS, C, alloc_size);  
  printf("This may take a while!\n\n");

  // // Arrays on GPU global memory
  // float *d_A, *d_B, *d_C;

  // // Arrays on the host memory
  // float *h_A, *h_B, *h_C, *h_C_dev;

  // int i, errCount = 0, zeroCount = 0;
  // size_t allocSize = arrLen * arrLen * sizeof(float);

  // // printf("Length of the array = %d\n", arrLen);

  // // Allocate GPU memory
  // CUDA_SAFE_CALL(cudaMalloc((void **)&d_A, allocSize));
  // CUDA_SAFE_CALL(cudaMalloc((void **)&d_B, allocSize));
  // CUDA_SAFE_CALL(cudaMalloc((void **)&d_C, allocSize));

  // // Allocate arrays on host memory
  // h_A                    = (float *) malloc(allocSize);
  // h_B                    = (float *) malloc(allocSize);
  // h_C                    = (float *) malloc(allocSize);
  // h_C_dev                = (float *) malloc(allocSize);
  // memset(h_C, 0, allocSize);

  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf(" OPT %d, iter %ld, size %ld\n", OPTION, x, n);
    set_matrix_row_length(a0, n);
    set_matrix_row_length(b0, n);
    set_matrix_row_length(c0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    mmm_ijk(a0, b0, c0, 4);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;

  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf(" OPT %d, iter %ld, size %ld\n", OPTION, x, n);
    set_matrix_row_length(a0, n);
    set_matrix_row_length(b0, n);
    set_matrix_row_length(c0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    mmm_ijk(a0, b0, c0, 64);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }
  
  return 0;
}

void init_matrix(float *mat, int len, int seed) {
  int i;
  int max_num = 100; // changes this to initialize with higher numbers
  // float randNum;
  // srand(seed);

  for (i = 0; i < len*len; i++) {
    // randNum = (float) rand() / RAND_MAX; // let randNum be a random integer
    // mat[i] = randNum;
    mat[i] = float(rand() % (max_num - 1));
  }
}

void print_array(float* v, int arr_len) {
  for (int i=0; i < arr_len; i++) {
    for (int j=0; j < arr_len; j++) {
      printf("%.3f ", v[i*arr_len + j]);
    }
    printf("\n");
  }
  printf("\n");
}