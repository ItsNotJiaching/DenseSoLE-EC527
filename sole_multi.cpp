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
 * OpenMP Blocked LU Factorization
 * * @param A pointer to input matrix, the A in the Ax=b.
 * @param x pointer to output vector, the x in the Ax=b.
 * @param b pointer to the b in the Ax=b.
 * @param row_len Dimension of the square matrix
 */
void sole_omp(data_t* A, data_t* x, data_t* b, int row_len) {
    int B = 32; // Block size
    int N = row_len / B;

    // Part 1: Block LU Factorization
    for (int k = 0; k < N; k++) {
        
        // Step A: Factorize diagonal block (SERIAL)
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

        // Step B: Update block row (STATIC)
        #pragma omp parallel for schedule(static)
        for (int j = k + 1; j < N; j++) {
            for (int kk = k * B; kk < k * B + B; kk++) {
                for (int i = kk + 1; i < k * B + B; i++) {
                    for (int jj = j * B; jj < j * B + B; jj++) {
                        A[i * row_len + jj] -= A[i * row_len + kk] * A[kk * row_len + jj];
                    }
                }
            }
        }

        // Step C: Update block column (STATIC)
        #pragma omp parallel for schedule(static)
        for (int i = k + 1; i < N; i++) {
            for (int kk = k * B; kk < k * B + B; kk++) {
                data_t reciprocal = 1.0 / A[kk * row_len + kk];
                for (int ii = i * B; ii < i * B + B; ii++) {
                    A[ii * row_len + kk] *= reciprocal;
                    for (int j = kk + 1; j < k * B + B; j++) {
                        A[ii * row_len + j] -= A[ii * row_len + kk] * A[kk * row_len + j];
                    }
                }
            }
        }

        // Step D: Schur Complement Update (STATIC - Major Bottleneck)
        #pragma omp parallel for schedule(static)
        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++) {
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

    // Part 2: Substitution (SERIAL)
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

void sole_omp_balanced(data_t* A, data_t* x, data_t* b, int row_len) {
    int B = 32; // Block size
    int N = row_len / B;

    // Part 1: Block LU Factorization
    for (int k = 0; k < N; k++) {
        
        // Step A: Factorize diagonal block (SERIAL)
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

        // Step B: Update block row (MANUAL ATOMIC QUEUE)
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

        // Step C: Update block column (MANUAL ATOMIC QUEUE)
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

        // Step D: Schur Complement Update (MANUAL ATOMIC QUEUE)
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

    // Part 2: Substitution (SERIAL)
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
