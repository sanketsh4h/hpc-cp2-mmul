#include <cblas.h>

const char* dgemm_desc = "Blocked dgemm with copy optimization.";

// Define a macro for the block size (adjust as needed)
#define BLOCK_SIZE 32

/*
 * This routine performs a dgemm operation
 * C := C + A * B
 * where A, B, and C are n-by-n matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */
void square_dgemm(int n, double* A, double* B, double* C) 
{
    // Allocate temporary storage for blocks
    double* A_block = new double(BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    double* B_block = new double(BLOCK_SIZE * BLOCK_SIZE * sizeof(double));

    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            for (int k = 0; k < n; k += BLOCK_SIZE) {
                // Determine the dimensions of the current block
                int block_size_i = (i + BLOCK_SIZE <= n) ? BLOCK_SIZE : (n - i);
                int block_size_j = (j + BLOCK_SIZE <= n) ? BLOCK_SIZE : (n - j);
                int block_size_k = (k + BLOCK_SIZE <= n) ? BLOCK_SIZE : (n - k);

                // Copy A and B blocks into local storage
                for (int ii = 0; ii < block_size_i; ii++) {
                    for (int kk = 0; kk < block_size_k; kk++) {
                        A_block[ii * block_size_k + kk] = A[(i + ii) + (k + kk) * n];
                    }
                }

                for (int kk = 0; kk < block_size_k; kk++) {
                    for (int jj = 0; jj < block_size_j; jj++) {
                        B_block[kk * block_size_j + jj] = B[(k + kk) + (j + jj) * n];
                    }
                }

                // Perform matrix multiplication on the blocks
                for (int ii = 0; ii < block_size_i; ii++) {
                    for (int jj = 0; jj < block_size_j; jj++) {
                        for (int kk = 0; kk < block_size_k; kk++) {
                            C[(i + ii) + (j + jj) * n] += A_block[ii * block_size_k + kk] * B_block[kk * block_size_j + jj];
                        }
                    }
                }
            }
        }
    }

}
