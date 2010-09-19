#ifndef UTPM_BLAS_LAPACK_H
#define UTPM_BLAS_LAPACK_H

#include <cblas.h>
#include <clapack.h>

int utpm_daxpy(int P, int D, int N, double alpha, double *x, int incx, double *y, int incy);

int utpm_dscal(int P, int D, int N, double alpha, double *X, int incX);

int utpm_dgemm(int P, int D, enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, int M, int N,
                 int K, double alpha, double *A,
                 int lda, double *B, int ldb,
                 double beta, double *C, int ldc);

int utpm_dgemm_residual(int P, int D, int d, enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, int M, int N,
                 int K, double alpha, double *A,
                 int lda, double *B, int ldb,
                 double beta, double *C, int ldc);

int utpm_dgesv(int P, int D, enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                  int N, int NRHS,
                  double *A, int lda, int *ipiv,
                  double *B, int ldb);



#endif
