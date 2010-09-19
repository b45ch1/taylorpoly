#ifndef UTPM_H
#define UTPM_H

#include "utpm_helper_functions.h"
#include "utpm_blas_lapack.h"

int utpm_imul(int P, int D, int M, int N, double *y, int ldy, double *x, int ldx);
int utpm_dot(int P, int D, int M, int N, int K, double alpha, double *A, int *Astrides, double *B, int *Bstrides, double beta, double *C, int *Cstrides);
int utpm_solve(int P, int D, int N, int NRHS, int *ipiv, double *A, int *Astrides, double *B, int *Bstrides);
int utpm_lu(int P, int D, int N, int *ipiv, double *A, int *Astrides, double *work);

#endif
