#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "sole.cuh"

/* -=-=-=-=- Time measurement by clock_gettime() -=-=-=-=- */
double measure(struct timespec start, struct timespec end)
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

void detect_threads_setting() {
    long int i, ognt;
    char * env_ONT;

    /* Find out how many threads OpenMP thinks it is wants to use */
    #pragma omp parallel for
    for(i=0; i<1; i++) {
    ognt = omp_get_num_threads();
    }

    printf("omp's default number of threads is %d\n", ognt);

    /* If this is illegal (0 or less), default to a DEFAULT_THREADS value */
    int DEFAULT_THREADS = 2;
    if (ognt <= 0) {
        if (DEFAULT_THREADS != ognt) {
            printf("Using pre-set DEFAULT_THREADS value %d\n", DEFAULT_THREADS);
            ognt = DEFAULT_THREADS;
        }
    }

    omp_set_num_threads(ognt);

    /* Once again ask OpenMP how many threads it is going to use */
    #pragma omp parallel for
    for(i=0; i<1; i++) {
        ognt = omp_get_num_threads();
    }
    printf("Using %d threads for OpenMP\n", ognt);
}

/**
 * OpenMP Version of the base serial code (sole_serial) using Static Load Assignment
 * @author Jiaxing Wang
 */
void sole_omp_naive(data_t* A, data_t* x, data_t* b, int row_len) {
    // Record LU Decomposition Only time
    struct timespec time_start, time_stop;
    clock_gettime(CLOCK_MONOTONIC, &time_start);

    // LU Decomposition
    #pragma omp parallel
    {
        data_t reciprocal;
        for (int k = 0; k < row_len; k++) {
            reciprocal = 1/A[k*row_len + k];

            // Compute multipliers and store in lower triangle (L) 
            #pragma omp for schedule(static)
            for (int j = k + 1; j < row_len; j++) {
                A[j*row_len + k] *= reciprocal;                  
            }

            // updating trailing submatrix (schular)
            #pragma omp for schedule(static)
            for (int i = k + 1; i < row_len; i++) {
                for (int j = k + 1; j < row_len; j++) {
                    A[i*row_len + j] -= A[i*row_len + k] * A[k*row_len + j]; //A[i][j] = A[i][j] - A[i][k] * A[k][i]
                }
            }
        }
    }
    clock_gettime(CLOCK_REALTIME, &time_stop);
    printf("Time spent computing LU Decomposition: %.3f ms\n", 1.0e3 * interval(time_start, time_stop));
    clock_gettime(CLOCK_REALTIME, &time_start);

    // forward sub Ly = b (uses x instead of y for better spatial locality)
    for (int i = 0; i < row_len; i++) {
        data_t* row = &A[i * row_len];
        data_t sum = 0.0; //intermediatary sum for dot product
        #pragma omp parallel for reduction(-:sum)
        for (int j = 0; j < i; j++) //this basically creates L staircase
            sum += row[j] * x[j]; //lower half, basically. y[i] = b[i] - A[i*row_len] * y[j]. 
        x[i] = b[i] - sum;
        // x[i] /= 1.0 //divide by diagonal per formula. For lower diagonal, it's always one, so no point of calculating.
    }

    //back sub Ux = y
    for (int i = row_len - 1; i >= 0; i--) {
        data_t* row = &A[i * row_len];
        data_t sum = 0.0; //intermediatary sum for dot product
        #pragma omp parallel for reduction(-:sum)
        for (int j = i + 1; j < row_len; j++) //get ahead of diagonal to iterate through U
            sum += row[j] * x[j]; //upper half, basically. x[i] = y[i] - A[i*row_len] * x[j]. 
        x[i] = x[i] - sum; //writing existing y[i] into actual x[i]
        x[i] = x[i]/row[i]; //divide by diagonal per the formula 
    }
    clock_gettime(CLOCK_REALTIME, &time_stop);
    printf("Time spent computing subs: %.3f ms\n", 1.0e3 * interval(time_start, time_stop));
}

/**
 * OpenMP Version of the base serial code (sole_serial) using Static Load Assignment, But doing an "inverse load"
 * Group from outside in i.e. (first thread and last thread in one cluster, second thread and second to last thread, etc.)
 * @author Owen Jiang
 */
void sole_omp_altload(data_t* A, data_t* x, data_t* b, int row_len) {
    // LU Decomposition
    timespec time_start, time_stop, time_stamp;
    clock_gettime(CLOCK_REALTIME, &time_start);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads(); 

        data_t reciprocal;
        for (int k = 0; k < row_len; k++) {
            reciprocal = 1/A[k*row_len + k];

            // Compute multipliers and store in lower triangle (L) 
            #pragma omp for schedule(static)
            for (int j = k + 1; j < row_len; j++) {
                A[j*row_len + k] *= reciprocal;                  
            }

            int remaining_rows = row_len - (k + 1);

           #pragma omp for schedule(static)
            for (int m = 0; m < remaining_rows; m++) { //Gaussian 
                int i;
                // if m is even take from top, if m is odd take from bottom.
                if (m % 2 == 0) {
                    i = (k + 1) + (m / 2);
                } else {
                    i = (row_len - 1) - (m / 2);
                }
                // Standard inner loop update
                data_t* row_i = &A[i * row_len];
                data_t* row_k = &A[k * row_len];
                data_t multiplier = row_i[k];

                for (int j = k + 1; j < row_len; j++) {
                    row_i[j] -= multiplier * row_k[j];
                }
            }
        }
    }
    clock_gettime(CLOCK_REALTIME, &time_stop);
    printf("Time spent computing in OpenMP: %f\n", 1.0e3 * interval(time_start, time_stop));
    clock_gettime(CLOCK_REALTIME, &time_start);
    // forward sub Ly = b (uses x instead of y for better spatial locality)
    for (int i = 0; i < row_len; i++) {
        data_t* row = &A[i * row_len];
        data_t sum = 0.0; //intermediatary sum for dot product
        #pragma omp parallel for reduction(-:sum)
        for (int j = 0; j < i; j++) //this basically creates L staircase
            sum += row[j] * x[j]; //lower half, basically. y[i] = b[i] - A[i*row_len] * y[j]. 
        x[i] = b[i] - sum;
        // x[i] /= 1.0 //divide by diagonal per formula. For lower diagonal, it's always one, so no point of calculating.
    }

    //back sub Ux = y
    for (int i = row_len - 1; i >= 0; i--) {
        data_t* row = &A[i * row_len];
        data_t sum = 0.0; //intermediatary sum for dot product
        #pragma omp parallel for reduction(-:sum)
        for (int j = i + 1; j < row_len; j++) //get ahead of diagonal to iterate through U
            sum += row[j] * x[j]; //upper half, basically. x[i] = y[i] - A[i*row_len] * x[j]. 
        x[i] = x[i] - sum; //writing existing y[i] into actual x[i]
        x[i] = x[i]/row[i]; //divide by diagonal per the formula 
    }
    clock_gettime(CLOCK_REALTIME, &time_stop);
    printf("Time spent computing subs: %f\n", 1.0e3 * interval(time_start, time_stop));
}

void sole_omp_balanced(data_t* A, data_t* x, data_t* b, int row_len) {
    int B = 32; // Block size
    int N = row_len / B;

    // Block LU Decomposition
    for (int k = 0; k < N; k++) {
        
        // LU Decomposition for diagonal blocks
        for (int kk = k * B; kk < k * B + B; kk++) {
            data_t reciprocal = 1.0 / A[kk * row_len + kk];
            for (int i = kk + 1; i < k * B + B; i++) {
                A[i * row_len + kk] *= reciprocal;
            }
            for (int i = kk + 1; i < k * B + B; i++) {
                for (int j = kk + 1; j < k * B + B; j++) {
                    A[i * row_len + j] -= A[i * row_len + kk] * A[kk * row_len + j];
                }
            }
        }

        // Update block row (MANUAL ATOMIC QUEUE)
        int shared_j_counter = k + 1;
        #pragma omp parallel 
        {
            int my_j;
            while (true) {
                // Ticket Dispenser
                #pragma omp atomic capture
                {
                    my_j = shared_j_counter;
                    shared_j_counter++;
                }
                
                if (my_j >= N) break; // Queue is empty, thread exits loop

                for (int kk = k * B; kk < k * B + B; kk++) {
                    for (int i = kk + 1; i < k * B + B; i++) {
                        for (int jj = my_j * B; jj < my_j * B + B; jj++) {
                            A[i * row_len + jj] -= A[i * row_len + kk] * A[kk * row_len + jj];
                        }
                    }
                }
            }
        }

        // Update block column (MANUAL ATOMIC QUEUE)
        int shared_i_counter_c = k + 1;
        #pragma omp parallel 
        {
            int my_i;
            while (true) {
                // Ticket Dispenser
                #pragma omp atomic capture
                {
                    my_i = shared_i_counter_c;
                    shared_i_counter_c++;
                }
                
                if (my_i >= N) break;

                for (int kk = k * B; kk < k * B + B; kk++) {
                    data_t reciprocal = 1.0 / A[kk * row_len + kk];
                    for (int ii = my_i * B; ii < my_i * B + B; ii++) {
                        A[ii * row_len + kk] *= reciprocal;
                        for (int j = kk + 1; j < k * B + B; j++) {
                            A[ii * row_len + j] -= A[ii * row_len + kk] * A[kk * row_len + j];
                        }
                    }
                }
            }
        }

        // Schur Complement Update (MANUAL ATOMIC QUEUE)
        int shared_i_counter_d = k + 1;
        #pragma omp parallel 
        {
            int my_i;
            while (true) {
                // Ticket Dispenser
                #pragma omp atomic capture
                {
                    my_i = shared_i_counter_d;
                    shared_i_counter_d++;
                }
                
                if (my_i >= N) break;

                for (int j = k + 1; j < N; j++) {
                    for (int ii = my_i * B; ii < my_i * B + B; ii++) {
                        for (int kk = k * B; kk < k * B + B; kk++) {
                            data_t temp = A[ii * row_len + kk];
                            for (int jj = j * B; jj < j * B + B; jj++) {
                                A[ii * row_len + jj] -= temp * A[kk * row_len + jj];
                            }
                        }
                    }
                }
            }
        }
    }

    // Forward and backward sub (SERIAL)
    for (int i = 0; i < row_len; i++) {
        data_t* row = &A[i * row_len];
        data_t sum = 0.0; 
        for (int j = 0; j < i; j++) sum += row[j] * x[j]; 
        x[i] = b[i] - sum;
    }
    for (int i = row_len - 1; i >= 0; i--) {
        data_t* row = &A[i * row_len];
        data_t sum = 0.0; 
        for (int j = i + 1; j < row_len; j++) sum += row[j] * x[j]; 
        x[i] = (x[i] - sum) / row[i]; 
    }
}
