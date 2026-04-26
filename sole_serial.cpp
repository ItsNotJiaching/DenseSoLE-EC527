#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "sole.cuh"

void print_matrix(data_t* mat, int row_len) {
  for (int i=0; i < row_len; i++) {
    for (int j=0; j < row_len; j++) {
      printf("%.3f ", mat[i*row_len + j]);
    }
    printf("\n");
  }
  printf("\n");
}

/**
 * Serial Baseline. For an array length n, the computational complexity of 
 * naive LU decomposition is O(n^3), and its dataset size grows at O(n^2).
 * (Culler 236)
 * 
 * @param A pointer to input matrix, the A in the Ax=b.
 * @param x pointer to output vector, the x in the Ax=b.
 * @param b pointer to the b in the Ax=b.
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
                for i ← k+1 to N-1 do  //for all rows below this diagonal block
                    for j ← k+1 to N-1 do //for all blocks in the corr. row 
                        Ai,j ← Ai,j - Ai,k* (Ak,j);    
        endfor
        endfor
        endfor
        endfor

        Note that this example is in-place, so L and U are combined into one matrix.
    */
    // printf("Original A Matrix: \n");
    // print_matrix(A, row_len);

    data_t reciprocal; //precalculate division for each lower triangle calculation instead of computing division every time
    for (int k = 0; k < row_len; k++) {
        reciprocal = 1/A[k*row_len + k];
        // Compute multipliers and store in lower triangle (L) 
        for (int j = k + 1; j < row_len; j++) {
            A[j*row_len + k] *= reciprocal;                  
        }

        // printf("Intermediate %d: \n", k);
        // print_matrix(A, row_len);

        // for all rows below diagonal (U)
        for (int i = k + 1; i < row_len; i++) {
            for (int j = k + 1; j < row_len; j++) {
                A[i*row_len + j] -= A[i*row_len + k] * A[k*row_len + j];     
            }
        }

        // printf("Iteration %d: \n", k);
        // print_matrix(A, row_len);
    }
    // printf("Updated Matrix: \n");
    // print_matrix(A, row_len);

    //forward sub Ly = b (uses x instead of y for better spatial locality)
    // L[i][0]*y[0] + L[i][1]*y[1] + ... + L[i][i]*y[i] = b[i], but done in reverse: 
    // y[i] = b[i] - L[i][0]*y[0] - L[i][1]*y[1] ... because we already calculated y[0] and y[1] in previous passes, we can do this.
    // see Golub & Van Loan Matrix Computations for full algorithm explanation
    for (int i = 0; i < row_len; i++) {
        data_t* row = &A[i * row_len];
        data_t sum = 0.0; //intermediatary sum for dot product
        for (int j = 0; j < i; j++) //this basically creates L staircase
            sum += row[j] * x[j]; //lower half, basically. y[i] = b[i] - A[i*row_len] * y[j]. 
        x[i] = b[i] - sum;
        // x[i] /= 1.0 //divide by diagonal per formula. For lower diagonal, it's always one, so no point of calculating.
    }

    //back sub Ux = y
    //U[i][i]*x[i] + U[i][i+1]*x[i+1] + ... + U[i][n]*x[n] = y[i], but done in reverse:  
    //x[i] = (y[i] - U[i][i+1]*x[i+1] - ... - U[i][n]*x[n]) / U[i][i]   ... because we already calculated y[0] and y[1] in previous passes, we can do this.
    // see Golub & Van Loan Matrix Computations for full algorithm explanation
    for (int i = row_len - 1; i >= 0; i--) {
        data_t* row = &A[i * row_len];
        data_t sum = 0.0; //intermediatary sum for dot product
        for (int j = i + 1; j < row_len; j++) //get ahead of diagonal to iterate through U
            sum += row[j] * x[j]; //upper half, basically. x[i] = y[i] - A[i*row_len] * x[j]. 
        x[i] = x[i] - sum; //writing existing y[i] into actual x[i]
        x[i] = x[i]/row[i]; //divide by diagonal per the formula 
    }
}

/**
 * Serial Blocking Optimization.
 * @param A pointer to input matrix, the A in the Ax=b.
 * @param x pointer to output vector, the x in the Ax=b.
 * @param b pointer to the b in the Ax=b.
 * @param B block size
 */
void sole_blocked(data_t* A, data_t* x, data_t* b, int row_len, int B) {
    int N = row_len / B; // N blocks in array
    // Part 1: Block LU Factorization
    for (int k = 0; k < N; k++) {
        // Step A: Factorize diagonal block A_{k, k} (Computes L_kk and U_kk in place)
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

        // Step B: Update block row (Computes U_{k, j} using forward substitution)
        for (int j = k + 1; j < N; j++) {
            for (int kk = k * B; kk < k * B + B; kk++) {
                for (int i = kk + 1; i < k * B + B; i++) {
                    for (int jj = j * B; jj < j * B + B; jj++) {
                        A[i * row_len + jj] -= A[i * row_len + kk] * A[kk * row_len + jj];
                    }
                }
            }
        }

        // Step C: Update block column (Computes L_{i, k} using backward substitution)
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

        // Step D: Schur Complement Update (A_{i, j} = A_{i, j} - L_{i, k} * U_{k, j})
        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++) {
                // Highly cache-friendly MMM loop (kij ordering)
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

    // Part 2: Forward and Backward Substitution
    // Same as in fully serial code
    // Forward substitution: Ly = b
    for (int i = 0; i < row_len; i++) {
        data_t* row = &A[i * row_len];
        data_t sum = 0.0; 
        for (int j = 0; j < i; j++) {
            sum += row[j] * x[j]; 
        } 
        x[i] = b[i] - sum;
    }

    // Backward substitution: Ux = y
    for (int i = row_len - 1; i >= 0; i--) {
        data_t* row = &A[i * row_len];
        data_t sum = 0.0; 
        for (int j = i + 1; j < row_len; j++) {
            sum += row[j] * x[j]; 
        }
        x[i] = x[i] - sum; 
        x[i] = x[i] / row[i]; 
    }
}