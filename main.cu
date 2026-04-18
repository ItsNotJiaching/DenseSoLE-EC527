#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <time.h>
#include "sole_multi.h"
#include "sole_serial.h"
#include "sole_gpu.cuh"

void init_matrix(data_t *mat, int len, int seed);
void print_array(data_t* v, int arr_len);


// Let's do multiples of 8 for now
#define A   8  /* coefficient of x^2 */
#define B   8  /* coefficient of x */
#define C   16  /* constant term */

#define NUM_TESTS 15   /* Number of different sizes to test */

#define OPTIONS 1

/* -=-=-=-=- Time measurement by clock_gettime() -=-=-=-=- */
/*
  As described in the clock_gettime manpage (type "man clock_gettime" at the
  shell prompt), a "timespec" is a structure that looks like this:
 
        struct timespec {
          time_t   tv_sec;   // seconds
          long     tv_nsec;  // and nanoseconds
        };
 */

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
/*
     This method does not require adjusting a #define constant

  How to use this method:

      struct timespec time_start, time_stop;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
      // DO SOMETHING THAT TAKES TIME
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
      measurement = interval(time_start, time_stop);

 */


/* -=-=-=-=- End of time measurement declarations =-=-=-=- */

/* This routine "wastes" a little time to make sure the machine gets
   out of power-saving mode (800 MHz) and switches to normal speed. */
double wakeup_delay()
{
  double meas = 0; int i, j;
  struct timespec time_start, time_stop;
  double quasi_random = 0;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
  j = 100;
  while (meas < 1.0) {
    for (i=1; i<j; i++) {
      /* This iterative calculation uses a chaotic map function, specifically
         the complex quadratic map (as in Julia and Mandelbrot sets), which is
         unpredictable enough to prevent compiler optimisation. */
      quasi_random = quasi_random*quasi_random - 1.923432;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    meas = interval(time_start, time_stop);
    j *= 2; /* Twice as much delay next time, until we've taken 1 second */
  }
  return quasi_random;
}



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
  data_t* arrA = (data_t *) malloc(alloc_size_mat);
  data_t* arrX = (data_t *) malloc(alloc_size_vec);
  data_t* arrB = (data_t *) malloc(alloc_size_vec);

  // Terminal Output
  printf("Running Dense System of Linear Equations (SoLE) Test ");
  printf("for %d different matrix sizes from %d to %d\n\n", NUM_TESTS, C, arr_len);  
  printf("This may take a while!\n\n");

  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=arr_len); x++) {
    init_matrix(arrA, n, 42);
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

  printf("row_len, serial_naive, serial_blocked\n");
  {
    int i, j;
    for (i = 0; i < NUM_TESTS; i++) {
      printf("%ld, ", A*i*i + B*i + C);
      for (j = 0; j < OPTIONS; j++) {
        if (j != 0) {
          printf(", ");
        }
        printf("%ld", (long int) (1e9 * time_stamp[j][i]));
      }
      printf("\n");
    }
  }
  printf("\n");

  printf("Wakeup delay computed: %g \n", wakeup_answer);
  return 0;
}

void init_matrix(data_t *mat, int len, int seed) {
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

void print_array(data_t* v, int arr_len) {
  for (int i=0; i < arr_len; i++) {
    for (int j=0; j < arr_len; j++) {
      printf("%.3f ", v[i*arr_len + j]);
    }
    printf("\n");
  }
  printf("\n");
}