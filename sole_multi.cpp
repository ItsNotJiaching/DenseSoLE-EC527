#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "sole.cuh"

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
    // struct timespec time_start, time_stop;
    // clock_gettime(CLOCK_MONOTONIC, &time_start);

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
    // clock_gettime(CLOCK_MONOTONIC, &time_stop);
    // printf("Time spent computing LU Decomposition: %.3f ms\n", 1.0e3 * interval(time_start, time_stop));
    // clock_gettime(CLOCK_MONOTONIC, &time_start);

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
    // clock_gettime(CLOCK_REALTIME, &time_stop);
    // printf("Time spent computing subs: %.3f ms\n", 1.0e3 * interval(time_start, time_stop));
}

/**
 * OpenMP Version of the base serial code (sole_serial) using Static Load Assignment, But doing an "inverse load"
 * Group from outside in i.e. (first thread and last thread in one cluster, second thread and second to last thread, etc.)
 * @author Owen Jiang
 */
void sole_omp_altload(data_t* A, data_t* x, data_t* b, int row_len) {
    // LU Decomposition
    // timespec time_start, time_stop, time_stamp;
    // clock_gettime(CLOCK_REALTIME, &time_start);
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
    // clock_gettime(CLOCK_REALTIME, &time_stop);
    // printf("Time spent computing in OpenMP: %f\n", 1.0e3 * interval(time_start, time_stop));
    // clock_gettime(CLOCK_REALTIME, &time_start);
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
    // clock_gettime(CLOCK_REALTIME, &time_stop);
    // printf("Time spent computing subs: %f\n", 1.0e3 * interval(time_start, time_stop));
}

/**
 * OpenMP Version with load balancing
 * Used collapse(2) in LU Decomposition to merge schular submatrix
 * Optimized small loops handling and shared sum management w/ explicit reset via single
 * @author Alvin Yan
 */
void sole_omp_optimized(data_t* A, data_t* x, data_t* b, int row_len) {
    // Record LU Decomposition Only time
    // struct timespec time_start, time_stop;
    // clock_gettime(CLOCK_MONOTONIC, &time_start);

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
            #pragma omp for schedule(static) collapse(2)
            // collapse(2) merges i and j loops into one workload, reduces underutilization
            // k=row_len-2: 1×1 = 1   unit  --> 1 thread works,  7 idle 
            // k=row_len-4: 3×3 = 9   units --> 8 threads work,  0 idle
            // k=row_len-9: 8×8 = 64  units --> 8 threads work,  0 idle
            for (int i = k + 1; i < row_len; i++) {
                for (int j = k + 1; j < row_len; j++) {
                    A[i*row_len + j] -= A[i*row_len + k] * A[k*row_len + j]; //A[i][j] = A[i][j] - A[i][k] * A[k][i]
                }
            }
        }
    }
    // clock_gettime(CLOCK_MONOTONIC, &time_stop);
    // printf("Time spent computing LU Decomposition: %.3f ms\n", 1.0e3 * interval(time_start, time_stop));
    // clock_gettime(CLOCK_MONOTONIC, &time_start);

    int nthreads = omp_get_max_threads();
    const int THRESHOLD = nthreads * 2;

    // Shared accumulator for reductions inside the persistent region
    // Must be zeroed before each row's reduction
    data_t sum = 0.0;

    #pragma omp parallel shared(sum)
    {
        // Forward substitution: Ly = b
        for (int i = 0; i < row_len; i++) {
            data_t* row = &A[i * row_len];

            if (i < THRESHOLD) {
                #pragma omp single
                {
                    data_t s = 0.0;
                    for (int j = 0; j < i; j++) s += row[j] * x[j];
                    x[i] = b[i] - s;
                }
            } else {
                // Reset before reduction: reduction(+:sum) adds privates INTO
                // sum_initial, so without this reset previous row's value accumulates
                #pragma omp single
                sum = 0.0;

                // reduction(+:sum): private_i initialized to 0, accumulates partial
                // sum with +=, combines as sum = 0 + p0 + p1 + ... = true_sum
                #pragma omp for reduction(+:sum)
                for (int j = 0; j < i; j++)
                    sum += row[j] * x[j];  // ← += throughout, no sign confusion

                #pragma omp single
                x[i] = b[i] - sum;  // b[i] - true_sum  ← correct
            }
        }

        // Backward substitution: Ux = y
        for (int i = row_len - 1; i >= 0; i--) {
            data_t* row = &A[i * row_len];
            int remaining = row_len - 1 - i;

            if (remaining < THRESHOLD) {
                #pragma omp single
                {
                    data_t s = 0.0;
                    for (int j = i + 1; j < row_len; j++) s += row[j] * x[j];
                    x[i] = (x[i] - s) / row[i];
                }
            } else {
                #pragma omp single
                sum = 0.0;

                // reduction(+:sum): same logic as forward sub
                // sum += gives true_sum directly, subtraction happens outside
                #pragma omp for reduction(+:sum)
                for (int j = i + 1; j < row_len; j++)
                    sum += row[j] * x[j];  // ← += not -=

                #pragma omp single
                x[i] = (x[i] - sum) / row[i];  // (x[i] - true_sum) / diag  ← correct
            }
        }
    }

    // clock_gettime(CLOCK_MONOTONIC, &time_stop);
    // printf("Time spent computing subs: %.3f ms\n", 1.0e3 * interval(time_start, time_stop));
}