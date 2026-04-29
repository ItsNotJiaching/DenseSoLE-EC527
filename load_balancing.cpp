#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <cstring>
#include "sole.cuh"

void init_matrix(data_t *mat, int len);
void init_vector(data_t *mat, int len);
double verify(data_t* arrA, data_t* arrX, data_t* arrB, int n);


// Let's do multiples of 8 for now
#define A   0  /* coefficient of x^2 */
#define B   256  /* coefficient of x */
#define C   256  /* constant term */

#define NUM_TESTS 24   /* Number of different sizes to test */

#define OPTIONS 9
#define TOLERANCE 1e-4

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
   out of power-saving mode and switches to normal speed. */
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
int main() {
    // Initializations for time recording
    struct timespec time_start, time_stop;
    // We repurpose OPTIONS to represent different thread counts
    int thread_settings[] = {1, 2, 4, 8, 16, 24, 28}; 
    #define NUM_THREAD_SETTINGS 8
    
    double time_stamp[NUM_THREAD_SETTINGS][NUM_TESTS];
    double error[NUM_THREAD_SETTINGS][NUM_TESTS];
    double wakeup_answer = wakeup_delay();

    // Use a fixed block size to isolate the effect of thread scaling on load balance
    int FIXED_BLOCK_SIZE = 64; 
    int TILE_SIZE = 16;

    // Determine max array length
    long int x, n, arr_len;
    x = NUM_TESTS - 1;
    arr_len = A*x*x + B*x + C;

    // Initialize master data arrays
    data_t* arrA = (data_t *) calloc(arr_len * arr_len, sizeof(data_t));
    data_t* arrB = (data_t *) calloc(arr_len, sizeof(data_t));
    init_matrix(arrA, arr_len);
    init_vector(arrB, arr_len);

    printf("Running Thread Scaling Test (Fixed Block Size: %d)\n", FIXED_BLOCK_SIZE);
    printf("Testing matrix sizes from %d to %ld\n\n", C, arr_len);

    // Loop through our "Options" (which are now Thread Counts)
    for (int OPTION = 0; OPTION < NUM_THREAD_SETTINGS; OPTION++) {
        int current_threads = thread_settings[OPTION];
        omp_set_num_threads(current_threads);
        
        printf("Running OPTION %d: Threads = %d\n", OPTION, current_threads);

        for (x = 0; x < NUM_TESTS; x++) {
            n = A*x*x + B*x + C;
            if (n > arr_len) break;

            // Copy originals to ensure numerical fresh start
            data_t* arrA_copy = (data_t*) malloc(n * n * sizeof(data_t));
            data_t* arrB_copy = (data_t*) malloc(n * sizeof(data_t));
            data_t* arrX = (data_t *) calloc(n, sizeof(data_t));
            
            memcpy(arrA_copy, arrA, n * n * sizeof(data_t));
            memcpy(arrB_copy, arrB, n * sizeof(data_t));

            printf("  Size %ld...\n", n);
            
            clock_gettime(CLOCK_MONOTONIC, &time_start);
            
            // Execute the blocked solver
            sole_omp_tiled_unrolling(arrA_copy, arrX, arrB_copy, n, FIXED_BLOCK_SIZE, TILE_SIZE);
            
            clock_gettime(CLOCK_MONOTONIC, &time_stop);
            
            time_stamp[OPTION][x] = interval(time_start, time_stop);
            error[OPTION][x] = verify(arrA, arrX, arrB, n);

            // Error Prompt: If the result is mathematically wrong, alert the user
            if (error[OPTION][x] > (TOLERANCE * n)) {
                fprintf(stderr, "!! MATH ERROR at Threads: %d, Size: %ld, Error: %e !!\n", 
                        current_threads, n, error[OPTION][x]);
            }

            free(arrA_copy); free(arrB_copy); free(arrX);
        }
    }

    printf("row_len");
    for(int j=0; j < NUM_THREAD_SETTINGS; j++) {
        printf(", threads_%d, err_%d", thread_settings[j], thread_settings[j]);
    }
    printf("\n");

    for (int i = 0; i < NUM_TESTS; i++) {
        printf("%ld, ", A*i*i + B*i + C);
        for (int j = 0; j < NUM_THREAD_SETTINGS; j++) {
            if (j != 0) printf(", ");
            // Printing in Nanoseconds to match your original script
            printf("%.1f, %.2f", (double)(1e9 * time_stamp[j][i]), error[j][i]);
        }
        printf("\n");
    }

    printf("Time units: ns\n");
    printf("Wakeup delay computed: %g \n", wakeup_answer);

    free(arrA); free(arrB);
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
 * Computes the total amount of error that is ABOVE tolerance
 * @returns The total amount of error that is ABOVE tolerance
 * @authors Owen Jiang, Jiaxing Wang
 */
double verify(data_t* arrA, data_t* arrX, data_t* arrB, int n) {
  double err = 0.0;
  for (int i = 0; i < n; i++) {
    double ax_i = 0.0;
    for (int j = 0; j < n; j++)
      ax_i += (double)arrA[i*n + j] * (double)arrX[j];
    double diff = fabs(ax_i - (double)arrB[i]);
    if (diff > TOLERANCE) err += diff;
  }
  return err;
}

/**
 * Computes max |Ax - b| using the original A and b. 
 * Expect some level of error when dealing with floating types,
 * but it should be consistent across implementations.
 * @returns The maximum error |Ax - b| of the data_t data type compared to double.
 * @author Owen Jiang
 */
/* Original Version */
// double verify(data_t* arrA, data_t* arrX, data_t* arrB, int n) {
//   double max_err = 0.0;
//   for (int i = 0; i < n; i++) {
//     double ax_i = 0.0;
//     for (int j = 0; j < n; j++)
//       ax_i += (double)arrA[i*n + j] * (double)arrX[j];
//     double err = fabs(ax_i - (double)arrB[i]);
//     if (err > max_err) max_err = err;
//   }
//   return max_err;
// }