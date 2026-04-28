#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cstring>
#include "sole.cuh"

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
 * Serial Blocking of LU Decomposition. Forward and backward passes are optimized
 * using multiple (4) accumulators. 
 * @param A pointer to input matrix, the A in the Ax=b.
 * @param x pointer to output vector, the x in the Ax=b.
 * @param b pointer to the b in the Ax=b.
 * @param row_len Row length of array (total size would be row_len^2)
 * @param B block size
 * 
 * @author Jiaxing Wang
 */
void sole_blocked(data_t* A, data_t* x, data_t* b, int row_len, int B) {
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

#include <immintrin.h>  // AVX + FMA intrinsics

/**
 * Reduces a __m256d holding [v0, v1, v2, v3] to a single double v0+v1+v2+v3.
 * Used to collapse a vectorized dot product accumulator into a scalar sum.
 *
 * Step 1: split 256-bit register into two 128-bit halves
 *         lo = [v0, v1],  hi = [v2, v3]
 * Step 2: add halves elementwise: sum2 = [v0+v2, v1+v3]
 * Step 3: shuffle to align the two sums, then add: (v0+v2) + (v1+v3)
 */
static inline double hsum256_pd(__m256d v) {
    __m128d lo   = _mm256_castpd256_pd128(v);    // [v0, v1]  (no instruction, free)
    __m128d hi   = _mm256_extractf128_pd(v, 1);  // [v2, v3]
    __m128d sum2 = _mm_add_pd(lo, hi);            // [v0+v2, v1+v3]
    __m128d shuf = _mm_unpackhi_pd(sum2, sum2);   // [v1+v3, v1+v3]
    return _mm_cvtsd_f64(_mm_add_sd(sum2, shuf)); // scalar: v0+v1+v2+v3
}

/**
 * AVX SIMD SoLE solver.
 * Direct vectorization of sole_serial without blocking.
 * Requires AVX + FMA (Haswell/2013+ on Intel, Excavator/2015+ on AMD).
 * data_t must be double (64-bit). For float, replace __m256d with __m256
 * and all _pd suffixes with _ps, giving 8 lanes instead of 4.
 * @param A pointer to input matrix, the A in the Ax=b.
 * @param x pointer to output vector, the x in the Ax=b.
 * @param b pointer to the b in the Ax=b.
 * @param row_len Row length of array (total size would be row_len^2)
 * 
 * @author Alvin Yan
 */
void sole_avx(data_t* A, data_t* x, data_t* b, int row_len) {
    for (int k = 0; k < row_len; k++) {
        data_t reciprocal = 1.0 / A[k * row_len + k];

        // Multiplier column: A[j*row_len + k] for j = k+1..row_len-1
        // These are STRIDED — each element is row_len doubles apart in memory.
        // AVX gather (_mm256_i64gather_pd) exists but the gather latency is
        // ~20 cycles on most CPUs vs ~5 for contiguous loads — not worth it
        // for a short loop. Scalar is cheaper here.
        for (int j = k + 1; j < row_len; j++)
            A[j * row_len + k] *= reciprocal;

        // Schur complement: A[i][j] -= A[i][k] * A[k][j]
        // Inner j-loop: A[i][j] and A[k][j] are CONTIGUOUS — AVX-friendly.
        // A[i][k] is scalar for a fixed i — broadcast once to all 4 lanes.
        data_t* row_k = &A[k * row_len]; // base pointer to pivot row (U)

        for (int i = k + 1; i < row_len; i++) {
            data_t* row_i = &A[i * row_len];

            // Hoist scalar multiplier out of the j-loop.
            // Store as both scalar (for tail) and SIMD broadcast (for main loop).
            data_t  mult_s = row_i[k];
            __m256d mult_v = _mm256_set1_pd(mult_s); // [mult, mult, mult, mult]

            int j = k + 1;

            // Main loop: 4 doubles per iteration.
            // _mm256_fnmadd_pd(a, b, c) = -(a*b) + c = c - a*b
            // One FMA instruction replaces: load, multiply, subtract, store.
            // Unaligned loads used since k+1 may not be 32-byte aligned.
            for (; j <= row_len - 4; j += 4) {
                __m256d aij = _mm256_loadu_pd(&row_i[j]); // A[i][j..j+3]
                __m256d akj = _mm256_loadu_pd(&row_k[j]); // A[k][j..j+3]
                // aij = aij - mult_v * akj
                aij = _mm256_fnmadd_pd(mult_v, akj, aij);
                _mm256_storeu_pd(&row_i[j], aij);
            }

            // Scalar tail: handles remaining 0..3 elements when
            // (row_len - k - 1) is not a multiple of 4
            for (; j < row_len; j++)
                row_i[j] -= mult_s * row_k[j];
        }
    }

    // Forward substitution: Ly = b
    // sum = dot(row[0..i-1], x[0..i-1]) — contiguous dot product, vectorizable.
    // Outer i-loop stays serial: x[i] depends on x[0..i-1].
    for (int i = 0; i < row_len; i++) {
        data_t* row  = &A[i * row_len];
        __m256d vsum = _mm256_setzero_pd(); // accumulator: [0, 0, 0, 0]

        int j = 0;
        // _mm256_fmadd_pd(a, b, c) = a*b + c
        // Accumulates 4 products into vsum per iteration.
        // Loop guard j <= i-4 ensures we never load past index i-1.
        for (; j <= i - 4; j += 4) {
            __m256d rj = _mm256_loadu_pd(&row[j]);
            __m256d xj = _mm256_loadu_pd(&x[j]);
            vsum = _mm256_fmadd_pd(rj, xj, vsum); // vsum += row[j..j+3] * x[j..j+3]
        }

        // Collapse 4-lane SIMD accumulator to scalar, then add remaining tail
        data_t sum = hsum256_pd(vsum);
        for (; j < i; j++) sum += row[j] * x[j]; // 0..3 remaining elements

        x[i] = b[i] - sum;
    }

    // Backward substitution: Ux = y
    // Mirror of forward sub — sum = dot(row[i+1..n-1], x[i+1..n-1])
    for (int i = row_len - 1; i >= 0; i--) {
        data_t* row  = &A[i * row_len];
        __m256d vsum = _mm256_setzero_pd();

        int j = i + 1;
        // Loop guard j <= row_len-4 prevents reading past end of row
        for (; j <= row_len - 4; j += 4) {
            __m256d rj = _mm256_loadu_pd(&row[j]);
            __m256d xj = _mm256_loadu_pd(&x[j]);
            vsum = _mm256_fmadd_pd(rj, xj, vsum);
        }

        data_t sum = hsum256_pd(vsum);
        for (; j < row_len; j++) sum += row[j] * x[j];

        x[i] = (x[i] - sum) / row[i];
    }
}