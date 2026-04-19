#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sole.cuh"

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
    data_t reciprocal; //precalculate division for each lower triangle calculation instead of computing division every time
    for (int k = 0; k < row_len; k++) {
        reciprocal = 1/A[k*row_len + k];
        // Compute multipliers and store in lower triangle (L) 
        for (int j = k + 1; j < row_len; j++) {
            A[j*row_len + k] *= reciprocal;                  
        }
        // for all rows below diagonal (U)
        for (int i = k + 1; i < row_len; i++) {
            for (int j = k + 1; j < row_len; j++) {
                A[i*row_len + j] -= A[i*row_len + k] * A[k*row_len + j];     
            }
        }
    }

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

data_t* transpose(data_t* matrix) {
    data_t* out;
    // get transpose code from lab 1 or 2
    return out;
}

/**
 * Serial Blocking Optimization.
 * 
 * @param A pointer to input matrix, the A in the Ax=b.
 * @param x pointer to output vector, the x in the Ax=b.
 * @param b pointer to the b in the Ax=b.
 * @param B block size
 * 
 * @return No return; computed outputs are stored in x.
 * @author Jiaxing Wang
 */
void sole_blocked(data_t* A, data_t* x, data_t* b, int row_len, int B) {
    printf("Running Baseline Blocked Code: \n");
    // Pseudocode
    /*
        for k = 0 to N-1 do // loop over all diagonal blocks
            decompose block A_{k, k}
            for j = k+1 to N-1 do // for all blocks in row & right of this diagonal block, do this
                A[k, j] = A[k, j] * A[k,k]^(-1) // divide by diagonal block
                for i = k+1 to N-1 do
                    for j2 = k+1 to N-1 do
                        A[i, j2] = A[i, j2] - A[i, k] * transpose(A[k, j])
    */ 
   for (int k=0; k < B; k++) {
        // decompose block A[k, k]
        for (int j=k+1; j < B; j++) {
            // A[k, j] = A[k, j] * A[k,k]^(-1) // divide by diagonal block
        }
   }
}

/*
Jiaxing Notes
 - blocking preserves memory locality (Culler 237)
 - interleaving & partitioning by scatter decomposition helps alleviates load balance (Culler 237)
    - DOES NOT SOLVE it though
 - test to find ideal block size; B=16, B=32 advised to be good for parallelism (Culler 238)

*/