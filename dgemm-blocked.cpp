const char* dgemm_desc = "Blocked dgemm.";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C) 
{
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int k = 0; k < n; k += block_size) {
                // Compute block boundaries
                int i_end = i + block_size < n ? i + block_size : n;
                int j_end = j + block_size < n ? j + block_size : n;
                int k_end = k + block_size < n ? k + block_size : n;

                // Perform matrix multiplication on blocks
                for (int ii = i; ii < i_end; ii++) {
                    for (int jj = j; jj < j_end; jj++) {
                        double cij = C[ii + jj * n];
                        for (int kk = k; kk < k_end; kk++) {
                            cij += A[ii + kk * n] * B[kk + jj * n];
                        }
                        C[ii + jj * n] = cij;
                    }
                }
            }
        }
    }
}
