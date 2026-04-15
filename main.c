
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include "sole_multi.h"
#include "sole_serial.h"

void initializeArray1D(float *arr, int len, int seed);
void print_array(float* v, int arr_len);


#define A   1  /* coefficient of x^2 */
#define B   1  /* coefficient of x */
#define C   2  /* constant term */

#define NUM_TESTS 60   /* Number of different sizes to test */

#define OPTIONS 3


int main(int argc, char **argv){

  printf("Running Dense System of Linear Equations (SoLE) Test!\n");

  int OPTION;
  struct timespec time_start, time_stop;
  double time_stamp[OPTIONS][NUM_TESTS];
  double wakeup_answer;
  long int x, n, alloc_size;

  x = NUM_TESTS-1;
  alloc_size = A*x*x + B*x + C;

  printf("Dense MMM tests \n\n");

  wakeup_answer = wakeup_delay();

  printf("Doing MMM three different ways,\n");
  printf("for %d different matrix sizes from %d to %d\n",
                                                     NUM_TESTS, C, alloc_size);
  printf("This may take a while!\n\n");

  /* declare and initialize the matrix structure */
  matrix_ptr a0 = new_matrix(alloc_size);
  init_matrix(a0, alloc_size);
  matrix_ptr b0 = new_matrix(alloc_size);
  init_matrix(b0, alloc_size);
  matrix_ptr c0 = new_matrix(alloc_size);
  zero_matrix(c0, alloc_size);

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
  float randNum;
  srand(seed);

  for (i = 0; i < len*len; i++) {
    // randNum = (float) rand() / RAND_MAX; // let randNum be a random integer
    randNum = float(rand() % (max_num - 1));
    mat[i] = randNum;
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