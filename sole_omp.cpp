#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "sole.cuh"

/**
 * From the OpenMP labs in EC527 to check how many threads are being used by the program.
 */
void detect_threads_setting() {
    long int i, ognt;
    char * env_ONT;
    /* Find out how many threads OpenMP thinks it is wants to use */
    #pragma omp parallel for
    for (i=0; i<1; i++) ognt = omp_get_num_threads();
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
    for (i=0; i<1; i++) ognt = omp_get_num_threads();
    printf("Using %d threads for OpenMP\n", ognt);
}

/**
 * Serial Baseline. For an array length n, the computational complexity of 
 * naive LU decomposition is O(n^3), and its dataset size grows at O(n^2).
 * (Culler 236)
 * 
 * @param A pointer to input matrix, the A in the Ax=b.
 * @param x pointer to output vector, the x in the Ax=b.
 * @param b pointer to the b in the Ax=b.
 * @param row_len Row length of array (total size would be row_len^2)
 * 
 * @return No return; computed outputs are stored in x.
 * @author Owen Jiang
 */
void sole_serial(data_t* A, data_t* x, data_t* b, int row_len) {

    /* 
        LU Factorization: Pseudocode
            for k ← 0 to N-1 do //loop over all diagonal blocks 
                for j ← k+1 to N-1 do  //for all blocks in the row of, and to the right of, this diagonal block 
                    Ak,j ← Ak,j * (Ak,k)-1; //divide by diagonal block 
                endfor
                for i ← k+1 to N-1 do  //for all rows below this diagonal block
                    for j ← k+1 to N-1 do //for all blocks in the corr. row 
                        Ai,j ← Ai,j - Ai,k* (Ak,j);    
                    endfor
                endfor
             endfor

        Note that this example is in-place, so L and U are combined into one matrix.
    */

    data_t reciprocal; // Calculate division outside j for loop
    for (int k = 0; k < row_len; k++) {
        reciprocal = 1/A[k*row_len + k];
        // Compute multipliers and store in lower triangle (L) 
        for (int j = k + 1; j < row_len; j++) {
            A[j*row_len + k] *= reciprocal;                  
        }

        // Compute row update for every row under and to right of pivot (U)
        for (int i = k + 1; i < row_len; i++) {
            for (int j = k + 1; j < row_len; j++) {
                A[i*row_len + j] -= A[i*row_len + k] * A[k*row_len + j];     
            }
        }
    }

    // Forward sub Ly = b 
    // Uses x instead of y for better spatial locality
    // L[i][0]*y[0] + L[i][1]*y[1] + ... + L[i][i]*y[i] = b[i], but done in reverse: 
    // y[i] = b[i] - L[i][0]*y[0] - L[i][1]*y[1] ... 
    // because we already calculated y[0] and y[1] in previous passes, we can do this.
    for (int i = 0; i < row_len; i++) {
        data_t* row = &A[i * row_len];
        data_t sum = 0.0;
        for (int j = 0; j < i; j++) // this basically creates L staircase
            sum += row[j] * x[j]; // lower half, basically. y[i] = b[i] - A[i*row_len] * y[j]. 
        x[i] = b[i] - sum;
    }

    // Back sub Ux = y
    // U[i][i]*x[i] + U[i][i+1]*x[i+1] + ... + U[i][n]*x[n] = y[i], but done in reverse:  
    // x[i] = (y[i] - U[i][i+1]*x[i+1] - ... - U[i][n]*x[n]) / U[i][i]   ... 
    // because we already calculated y[0] and y[1] in previous passes, we can do this.
    for (int i = row_len - 1; i >= 0; i--) {
        data_t* row = &A[i * row_len];
        data_t sum = 0.0;
        for (int j = i + 1; j < row_len; j++) // get ahead of diagonal to iterate through U
            sum += row[j] * x[j]; // upper half, basically. x[i] = y[i] - A[i*row_len] * x[j]. 
        x[i] = x[i] - sum; // writing existing y[i] into actual x[i]
        x[i] = x[i]/row[i]; // divide by diagonal per the formula 
    }
}

/**
 * OpenMP Version of the base serial code (sole_serial) using Static Load Assignment
 * @param A pointer to input matrix, the A in the Ax=b.
 * @param x pointer to output vector, the x in the Ax=b.
 * @param b pointer to the b in the Ax=b.
 * @param row_len row length of matrix
 * 
 * @return No return; computed outputs are stored in x.
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
    }

    // back sub Ux = y
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
 * @param A pointer to input matrix, the A in the Ax=b.
 * @param x pointer to output vector, the x in the Ax=b.
 * @param b pointer to the b in the Ax=b.
 * @param row_len row length of matrix
 * 
 * @return No return; computed outputs are stored in x.
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
 * OpenMP Version with attempted collapse optimization
 * Used collapse(2) in LU Decomposition to merge schular submatrix
 * Optimized small loops handling and shared sum management w/ explicit reset via single
 * @param A pointer to input matrix, the A in the Ax=b.
 * @param x pointer to output vector, the x in the Ax=b.
 * @param b pointer to the b in the Ax=b.
 * @param row_len Row length of array (total size would be row_len^2)
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

/**
 * OMP Naive Blocking of LU Decomposition. Forward and backward passes are optimized
 * using multiple (4) accumulators. 
 * @param A pointer to input matrix, the A in the Ax=b.
 * @param x pointer to output vector, the x in the Ax=b.
 * @param b pointer to the b in the Ax=b.
 * @param row_len Row length of array (total size would be row_len^2)
 * @param B block size
 * 
 * @author Owen Jiang, Jiaxing Wang
 */
void sole_omp_blocked(data_t* A, data_t* x, data_t* b, int row_len, int B) {
    int N = row_len / B; // N blocks in array
    // Blocked LU Decomposition
    // Much more complicated than serial LU decomposition!
    
    for (int k = 0; k < N; k++) {
        // LU Decomposition of diagonal block A[k, k], computed in-place
        for (int kk = k * B; kk < k * B + B; kk++) {
            data_t reciprocal = 1.0 / A[kk * row_len + kk];
            // Compute multipliers for the block
            for (int i = kk + 1; i < k * B + B; i++) {
                A[i * row_len + kk] *= reciprocal;
            }
            
            // Update the rest of the diagonal block
            for (int i = kk + 1; i < k * B + B; i++) {
                for (int j = kk + 1; j < k * B + B; j++) {
                    A[i * row_len + j] -= A[i * row_len + kk] * A[kk * row_len + j];
                }
            }
        }

        // Forward pass to update block row (computes U_{k, j})
        #pragma omp parallel for 
        for (int j = k + 1; j < N; j++) {
            for (int kk = k * B; kk < k * B + B; kk++) {
                for (int i = kk + 1; i < k * B + B; i++) {
                    for (int jj = j * B; jj < j * B + B; jj++) {
                        A[i * row_len + jj] -= A[i * row_len + kk] * A[kk * row_len + jj];
                    }
                }
            }
        }

        // Backward pass to update block column (computes L_{i, k}) AND
        // Schur Complement Update (A_{i, j} = A_{i, j} - L_{i, k} * U_{k, j})
        // Computing both under one i loop instead of splitting into two
        #pragma omp parallel for 
        for (int i = k + 1; i < N; i++) {
            // Backward Pass
            for (int kk = k * B; kk < k * B + B; kk++) {
                data_t reciprocal = 1.0 / A[kk * row_len + kk];
                for (int ii = i * B; ii < i * B + B; ii++) {
                    A[ii * row_len + kk] *= reciprocal;
                    for (int j = kk + 1; j < k * B + B; j++) {
                        A[ii * row_len + j] -= A[ii * row_len + kk] * A[kk * row_len + j];
                    }
                }
            }
            // Schur's Complement Update
            for (int j = k + 1; j < N; j++) {
                // kij ordering to optimize for cache
                for (int ii = i * B; ii < i * B + B; ii++) {
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

    // Forward substitution: Ly = b; same as Serial
    //#pragma omp parallel for
    for (int i = 0; i < row_len; i++) {
        data_t* row = &A[i * row_len];
        data_t sum = 0.0;
        data_t sum2 = 0;
        data_t sum3 = 0;
        data_t sum4 = 0;
        int j = 0;
        // Dot products using four accumulators
        for (j = 0; j < i-3; j+=4) {
            sum += row[j] * x[j]; 
            sum2 += row[j+1] * x[j+1];
            sum3 += row[j+2] * x[j+2];
            sum4 += row[j+3] * x[j+3];
        }
        // Finish the rest that accumulators didn't get
        for (; j < i; j++) sum += row[j] * x[j]; 
        // Changing associativity
        x[i] = b[i] - (sum + sum2) - (sum3 + sum4);
    }

    // Backward substitution: Ux = y; same as Serial
    //#pragma omp parallel for
    for (int i = row_len - 1; i >= 0; i--) {
        data_t* row = &A[i * row_len];
        data_t sum = 0.0; 
        data_t sum2 = 0;
        data_t sum3 = 0;
        data_t sum4 = 0;
        int j = 0;
        // Dot products using four accumulators
        for (j = i + 1; j < row_len-3; j+=4) {
            sum += row[j] * x[j]; 
            sum2 += row[j+1] * x[j+1];
            sum3 += row[j+2] * x[j+2];
            sum4 += row[j+3] * x[j+3];
        }
        // Finish the rest that accumulators didn't get
        for (; j < row_len; j++) sum += row[j] * x[j]; 
        // Changing associativity
        x[i] = x[i] - (sum + sum2) - (sum3 + sum4); 
        x[i] = x[i] / row[i]; 
    }
}

/**
 * OMP Load Balanced using OpenMP schedule dynamic and tiling
 * @param A pointer to input matrix, the A in the Ax=b.
 * @param x pointer to output vector, the x in the Ax=b.
 * @param b pointer to the b in the Ax=b.
 * @param row_len Row length of array (total size would be row_len^2)
 * @param B block size
 * @param T tile size
 * 
 * @author Owen Jiang, adapted from Jiaxing
 */
void sole_omp_tiled_unrolling(data_t* A, data_t* x, data_t* b, int row_len, int B, int T) {
    int N = row_len / B; // N blocks in array
    // Blocked LU Decomposition
    // Much more complicated than serial LU decomposition!
    
    for (int k = 0; k < N; k++) {
        // LU Decomposition of diagonal block A[k, k], computed in-place
        for (int kk = k * B; kk < k * B + B; kk++) {
            data_t reciprocal = 1.0 / A[kk * row_len + kk];
            // Compute multipliers for the block
            for (int i = kk + 1; i < k * B + B; i++) {
                A[i * row_len + kk] *= reciprocal;
            }
            
            // Update the rest of the diagonal block
            for (int i = kk + 1; i < k * B + B; i++) {
                for (int j = kk + 1; j < k * B + B; j++) {
                    A[i * row_len + j] -= A[i * row_len + kk] * A[kk * row_len + j];
                }
            }
        }

        // Forward pass to update block row (computes U_{k, j})
        #pragma omp parallel for 
        for (int j = k + 1; j < N; j++) {
            for (int kk = k * B; kk < k * B + B; kk++) {
                for (int i = kk + 1; i < k * B + B; i++) {
                    for (int jj = j * B; jj < j * B + B; jj++) {
                        A[i * row_len + jj] -= A[i * row_len + kk] * A[kk * row_len + jj];
                    }
                }
            }
        }

        // Backward pass to update block column (computes L_{i, k}) AND
        // Schur Complement Update (A_{i, j} = A_{i, j} - L_{i, k} * U_{k, j})
        // Computing both under one i loop instead of splitting into two
        #pragma omp parallel for 
        // Backward Pass
        for (int kk = k * B; kk < k * B + B; kk++) {
            data_t reciprocal = 1.0 / A[kk * row_len + kk];
            for (int ii = k * B; ii < k * B + B; ii++) {
                A[ii * row_len + kk] *= reciprocal;
                for (int j = kk + 1; j < k * B + B; j++) {
                    A[ii * row_len + j] -= A[ii * row_len + kk] * A[kk * row_len + j];
                }
            }
        }
        #pragma omp parallel for collapse(2) schedule(dynamic) //schedule dynamic has work pool where each thread updates their status
        // Idea is that tiling partitions work among each thread. Then, within each tile, block the matrix inside. This makes it easier to load balance
        // Schur's Complement Update
        for (int ii = k + 1; ii < N; ii += T) {
            for (int jj = k + 1; jj < N; jj += T) {

                for (int i = ii; i < ii + T && i < N; i++) {
                    for (int j = jj; j < jj + T && j < N; j++) {

                        // Block (i,j) update
                        for (int iii = i * B; iii < i * B + B; iii++) {
                            for (int kk = k * B; kk < k * B + B; kk++) {
                                data_t temp = A[iii * row_len + kk];
                                for (int jjj = j * B; jjj < j * B + B; jjj++) {
                                    A[iii * row_len + jjj] -= temp * A[kk * row_len + jjj];
                                }
                            }
                        }

                    }
                }

            }
        }
    }

    // Forward substitution: Ly = b; same as Serial
    //#pragma omp parallel for
    for (int i = 0; i < row_len; i++) {
        data_t* row = &A[i * row_len];
        data_t sum = 0.0;
        data_t sum2 = 0;
        data_t sum3 = 0;
        data_t sum4 = 0;
        int j = 0;
        // Dot products using four accumulators
        for (j = 0; j < i-3; j+=4) {
            sum += row[j] * x[j]; 
            sum2 += row[j+1] * x[j+1];
            sum3 += row[j+2] * x[j+2];
            sum4 += row[j+3] * x[j+3];
        }
        // Finish the rest that accumulators didn't get
        for (; j < i; j++) sum += row[j] * x[j]; 
        // Changing associativity
        x[i] = b[i] - (sum + sum2) - (sum3 + sum4);
    }

    // Backward substitution: Ux = y; same as Serial
    //#pragma omp parallel for
    for (int i = row_len - 1; i >= 0; i--) {
        data_t* row = &A[i * row_len];
        data_t sum = 0.0; 
        data_t sum2 = 0;
        data_t sum3 = 0;
        data_t sum4 = 0;
        int j = 0;
        // Dot products using four accumulators
        for (j = i + 1; j < row_len-3; j+=4) {
            sum += row[j] * x[j]; 
            sum2 += row[j+1] * x[j+1];
            sum3 += row[j+2] * x[j+2];
            sum4 += row[j+3] * x[j+3];
        }
        // Finish the rest that accumulators didn't get
        for (; j < row_len; j++) sum += row[j] * x[j]; 
        // Changing associativity
        x[i] = x[i] - (sum + sum2) - (sum3 + sum4); 
        x[i] = x[i] / row[i]; 
    }
}
