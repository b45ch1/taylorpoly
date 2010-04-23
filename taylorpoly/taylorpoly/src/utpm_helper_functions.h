#ifndef UTPM_HELPER_FUNCTIONS_H
#define UTPM_HELPER_FUNCTIONS_H


/* missing functions for element-wise matrix operations in BLAS/LAPACK */
inline int imul(int M, int N, double *y, int ldy, double *x, int ldx); 
inline int idiv(int M, int N, double *y, int ldy, double *x, int ldx);
inline int amul(int M, int N, double *x, int ldx, double *y, int ldy, double *z, int ldz);

/* helper functions */
inline int get_leadim_and_cblas_transpose(int M, int N, int *strides, int *leadim, int *trans);
inline int l_times_u(int N, double alpha, double *A, int lda, double *L, int ldl, double *U, int ldu);

#endif
