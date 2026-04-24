#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "sole.cuh"

void init_matrix(data_t *mat, int len);
void init_vector(data_t *mat, int len);
double verify(data_t* arrA, data_t* arrX, data_t* arrB, int n);
void print_array(data_t* v, int arr_len);


// Let's do multiples of 8 for now
#define A   32  /* coefficient of x^2 */
#define B   32  /* coefficient of x */
#define C   32  /* constant term */

#define NUM_TESTS 12   /* Number of different sizes to test */

#define OPTIONS 3

/* -=-=-=-=- Time measurement by clock_gettime() -=-=-=-=- */
double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}

/* This routine "wastes" a little time to make sure the machine gets
   out of power-saving mode (800 MHz) and switches to normal speed. */
double wakeup_delay()
{
  double meas = 0; int i, j;
  struct timespec time_start, time_stop;
  double quasi_random = 0;
  clock_gettime(CLOCK_MONOTONIC, &time_start);
  j = 100;
  while (meas < 1.0) {
    for (i=1; i<j; i++) {
      /* This iterative calculation uses a chaotic map function, specifically
         the complex quadratic map (as in Julia and Mandelbrot sets), which is
         unpredictable enough to prevent compiler optimisation. */
      quasi_random = quasi_random*quasi_random - 1.923432;
    }
    clock_gettime(CLOCK_MONOTONIC, &time_stop);
    meas = interval(time_start, time_stop);
    j *= 2; /* Twice as much delay next time, until we've taken 1 second */
  }
  return quasi_random;
}

/**
 * Main Function. Only testing square matrices with array sizes multiple of 32.
 */
int main(){
  // Initializations for time recording
  struct timespec time_start, time_stop;
  double time_stamp[OPTIONS][NUM_TESTS];
  double error[OPTIONS][NUM_TESTS];
  double wakeup_answer;
  wakeup_answer = wakeup_delay();

  // Initializations for determining array length
  long int x, n, arr_len;
  x = NUM_TESTS-1;
  arr_len = A*x*x + B*x + C;

  // Initalize data arrays to largest possible (to length arr_len)
  data_t* arrA = (data_t *) calloc(arr_len * arr_len, sizeof(data_t));
  data_t* arrB = (data_t *) calloc(arr_len, sizeof(data_t));
  init_matrix(arrA, arr_len);
  init_vector(arrB, arr_len);

  // Terminal Output
  printf("Running Dense System of Linear Equations (SoLE) Test ");
  printf("for %d different matrix sizes from %d to %d\n\n", NUM_TESTS, C, arr_len);
  detect_threads_setting();
  printf("Can declare max %d threads on this processor\n", omp_get_num_procs());
  printf("This may take a while!\n\n");

  int OPTION = 0;

  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=arr_len); x++) {
    //copy originals
    data_t* arrA_copy = (data_t*) calloc(arr_len * arr_len, sizeof(data_t)); // copy for error check
    data_t* arrB_copy = (data_t*) calloc(arr_len, sizeof(data_t)); // copy for error check
    memcpy(arrA_copy, arrA, n*n*sizeof(data_t));
    memcpy(arrB_copy, arrB, n*sizeof(data_t));
    data_t* arrX = (data_t *) calloc(arr_len, sizeof(data_t));

    printf(" Option %d, iter %ld, size %ld\n", OPTION, x, n);
    clock_gettime(CLOCK_MONOTONIC, &time_start);
    sole_omp_naive(arrA_copy, arrX, arrB_copy, n);
    clock_gettime(CLOCK_MONOTONIC, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
    error[OPTION][x] = verify(arrA, arrX, arrB,n);
  }

  OPTION++;
  
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=arr_len); x++) {
    //copy originals
    data_t* arrA_copy = (data_t*) calloc(arr_len * arr_len, sizeof(data_t)); // copy for error check
    data_t* arrB_copy = (data_t*) calloc(arr_len, sizeof(data_t)); // copy for error check
    memcpy(arrA_copy, arrA, n*n*sizeof(data_t));
    memcpy(arrB_copy, arrB, n*sizeof(data_t));
    data_t* arrX = (data_t *) calloc(arr_len, sizeof(data_t));

    printf(" Option %d, iter %ld, size %ld\n", OPTION, x, n);
    clock_gettime(CLOCK_MONOTONIC, &time_start);
    sole_omp_balanced(arrA_copy, arrX, arrB_copy, n);
    clock_gettime(CLOCK_MONOTONIC, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
    error[OPTION][x] = verify(arrA, arrX, arrB,n);
  }

  OPTION++;
  
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=arr_len); x++) {
    //copy originals
    data_t* arrA_copy = (data_t*) calloc(arr_len * arr_len, sizeof(data_t)); // copy for error check
    data_t* arrB_copy = (data_t*) calloc(arr_len, sizeof(data_t)); // copy for error check
    memcpy(arrA_copy, arrA, n*n*sizeof(data_t));
    memcpy(arrB_copy, arrB, n*sizeof(data_t));
    data_t* arrX = (data_t *) calloc(arr_len, sizeof(data_t));

    printf(" Option %d, iter %ld, size %ld\n", OPTION, x, n);
    clock_gettime(CLOCK_MONOTONIC, &time_start);
    sole_cuda(arrA_copy, arrX, arrB_copy, n, 1, 32);
    clock_gettime(CLOCK_MONOTONIC, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
    error[OPTION][x] = verify(arrA, arrX, arrB,n);
  }

  // OPTION++;
  
  // for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=arr_len); x++) {
  //   //copy originals
  //   data_t* arrA_copy = (data_t*) calloc(arr_len * arr_len, sizeof(data_t)); // copy for error check
  //   data_t* arrB_copy = (data_t*) calloc(arr_len, sizeof(data_t)); // copy for error check
  //   memcpy(arrA_copy, arrA, n*n*sizeof(data_t));
  //   memcpy(arrB_copy, arrB, n*sizeof(data_t));
  //   data_t* arrX = (data_t *) calloc(arr_len, sizeof(data_t));

  //   printf(" Option %d, iter %ld, size %ld\n", OPTION, x, n);
  //   clock_gettime(CLOCK_MONOTONIC, &time_start);
  //   sole_serial(arrA_copy, arrX, arrB_copy, n);
  //   clock_gettime(CLOCK_MONOTONIC, &time_stop);
  //   time_stamp[OPTION][x] = interval(time_start, time_stop);
  //   error[OPTION][x] = verify(arrA, arrX, arrB,n);
  // }

  // OPTION++;
  
  // for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=arr_len); x++) {
  //   //copy originals
  //   data_t* arrA_copy = (data_t*) calloc(arr_len * arr_len, sizeof(data_t)); // copy for error check
  //   data_t* arrB_copy = (data_t*) calloc(arr_len, sizeof(data_t)); // copy for error check
  //   memcpy(arrA_copy, arrA, n*n*sizeof(data_t));
  //   memcpy(arrB_copy, arrB, n*sizeof(data_t));
  //   data_t* arrX = (data_t *) calloc(arr_len, sizeof(data_t));

  //   printf(" Option %d, iter %ld, size %ld\n", OPTION, x, n);
  //   clock_gettime(CLOCK_MONOTONIC, &time_start);
  //   sole_blocked(arrA_copy, arrX, arrB_copy, n, 32);
  //   clock_gettime(CLOCK_MONOTONIC, &time_stop);
  //   time_stamp[OPTION][x] = interval(time_start, time_stop);
  //   error[OPTION][x] = verify(arrA, arrX, arrB,n);
  // }

  
  
  printf("row_len, omp_naive, omp_naive_error, "
        "omp_balanced, omp_balanced_error, "
        "cuda, cuda_error\n");
        // "serial_naive, serial_naive_error, "
        // "serial_blocked, serial_blocked_error\n");
  int i, j;
  for (i = 0; i < NUM_TESTS; i++) {
    printf("%ld, ", A*i*i + B*i + C);
    for (j = 0; j < OPTIONS; j++) {
      if (j != 0) {
        printf(", ");
      }
      printf("%.3f, %e", (double) (1e3 * time_stamp[j][i]), error[j][i]);
    }
    printf("\n");
  }
  printf("Time units: ms\n");

  printf("Wakeup delay computed: %g \n", wakeup_answer);
  return 0;
}

void init_matrix(data_t *mat, int len) {
  int i;
  int max_num = 100; // changes this to initialize with higher numbers
  // float randNum;
  // srand(seed);

  for (i = 0; i < len*len; i++) {
    // randNum = (float) rand() / RAND_MAX; // let randNum be a random integer
    // mat[i] = randNum;
    mat[i] = data_t(rand() % (max_num - 1));
  }
}

void init_vector(data_t *vec, int len) {
  int i;
  int max_num = 100; // changes this to initialize with higher numbers

  for (i = 0; i < len; i++)
    vec[i] = data_t(rand() % (max_num-1));
}

void print_array(data_t* v, int arr_len) {
  for (int i=0; i < arr_len; i++) {
    for (int j=0; j < arr_len; j++) {
      printf("%.3f ", v[i*arr_len + j]);
    }
    printf("\n");
  }
  printf("\n");
}

/**
 * Computes max |Ax - b| using the original A and b. 
 * Expect some level of error when dealing with floating types,
 * but it should be consistent across implementations.
 * @returns The maximum error |Ax - b| of the data_t data type compared to double.
 * @author Owen Jiang
 */
double verify(data_t* arrA, data_t* arrX, data_t* arrB, int n) {
  double max_err = 0.0;
  for (int i = 0; i < n; i++) {
    double ax_i = 0.0;
    for (int j = 0; j < n; j++)
      ax_i += (double)arrA[i*n + j] * (double)arrX[j];
    double err = fabs(ax_i - (double)arrB[i]);
    if (err > max_err) max_err = err;
  }
  return max_err;
}