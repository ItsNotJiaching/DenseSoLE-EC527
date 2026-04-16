
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

/**
 * Main Function. Only testing square matrices with array sizes multiple of 8.
 * (look into -- does SoLE apply to non-square matrices? Pretty sure no)
 */
int main(){
  int OPTION;
  struct timespec time_start, time_stop;
  double time_stamp[OPTIONS][NUM_TESTS];
  double wakeup_answer;
  long int x, n, arr_len;
  long int alloc_size_vec, alloc_size_mat;
  wakeup_answer = wakeup_delay();

  x = NUM_TESTS-1;
  arr_len = A*x*x + B*x + C;
  alloc_size_mat = arr_len * arr_len * sizeof(data_t);
  alloc_size_vec = arr_len * arr_len * sizeof(data_t);

  // Initalize data arrays
  float* arrA = (float *) malloc(alloc_size_mat);
  float* arrX;
  float* arrB;

  // Terminal Output
  printf("Running Dense System of Linear Equations (SoLE) Test ");
  printf("for %d different matrix sizes from %d to %d\n\n", NUM_TESTS, C, arr_len);  
  printf("This may take a while!\n\n");

  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=arr_len); x++) {
    printf(" OPT %d, iter %ld, size %ld\n", OPTION, x, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    sole_serial(arrA, arrX, arrB, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;

  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=arr_len); x++) {
    printf(" OPT %d, iter %ld, size %ld\n", OPTION, x, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    sole_blocked(arrA, arrX, arrB, n, 8); // Block Size 8
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