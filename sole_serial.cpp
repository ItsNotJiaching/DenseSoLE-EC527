#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sole_serial.h"

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
    printf("Running Baseline Serial Code: \n");
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