/*

Rationale:
----------

It turns out that it is a little inconvenient to provide UTPM algorithms of LAPACK and BLAS
since the API does not match the requirements very well. Also, many algorithms that are necessary are
completely missing.

The idea is to stay as close to BLAS and LAPACK as possible. The real algorithms in utpm.c may call
algorithms from this file. Please not that this file only provides a subset of the algorithms that
are necessary to do real computational work.

*/





#include <stdio.h>

#include "utpm_blas_lapack.h"


int utpm_dscal(int P, int D, int N, double alpha, double *X, int incX){
    cblas_dscal(N + (D-1)*P*N, alpha, X, incX);
    return 0;
}

int utpm_daxpy(int P, int D, int N, double alpha, double *x,
                 int incx, double *y, int incy){
    /*
    y = y + alpha * x
    
    See http://www.netlib.org/blas/daxpy.f for documentation.
    */
    
    /* input checks */
    if(x == y) return -1;
    
    cblas_daxpy(N + (D-1)*P*N, alpha, x, incx, y, incy);
    return 0;
}

int utpm_dgemm_residual(int p, int D, int d, enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, int M, int N,
                 int K, double alpha, double *A,
                 int lda, double *B, int ldb,
                 double beta, double *C, int ldc){

    /* Helper function to compute residuals of the form 
    \Delta F t^d =_{d+1} [Q^T]_d [Q]_d - Id
    
    for given p and d.
    
    computes:
    C = alpha \sum_{k=1}^{d-1} A_{d-k} B_k + beta C
    
    dimensions:
    -----------
    A_d is a (M,K) matrix
    B_d is a (K,N) matrix  for d = 0,...,D-1
    C is a (M,N) matrix
    
    See http://www.netlib.org/blas/dgemm.f for documentation.
    
    */
    
    int k;
    double *Ad, *Bd;
    double *Ap, *Bp;
    
    int pstrideA, pstrideB;
    int dstrideA, dstrideB;
    
    /* input checks */
    if(TransA != 111 || TransB != 111 || Order != 102){
        printf("The case TransA != 111 || TransB != 111 || Order != 101 has not been implemented yet!\n");
        return -1;
    }

    /* d > 0: higher order coefficients */
    dstrideA = lda*K; dstrideB = ldb*N; 
    pstrideA = (D-1)*dstrideA; pstrideB = (D-1)*dstrideB;
    
    Ap = A + p*pstrideA;
    Bp = B + p*pstrideB;
        
    /* compute sum_{k=1}^{d-1} x_k y_{d-k} */
    Ad = Ap + dstrideA;
    Bd = Bp + (d - 1) * dstrideB;
    for(k = 1; k < d; ++k){
        cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, Ad,
         lda, Bd, ldb, 1, C, ldc);
        Ad += dstrideA;
        Bd -= dstrideB;
    }
    return 0;
}
    
    

inline int utpm_dgemm(int P, int D, enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, int M, int N,
                 int K, double alpha, double *A,
                 int lda, double *B, int ldb,
                 double beta, double *C, int ldc){

    /* See http://www.netlib.org/blas/dgemm.f for documentation
    
    C = alpha A B + beta C
    
    dimensions:
    -----------
    A is a (M,K) matrix
    B is a (K,N) matrix
    C is a (M,N) matrix
    
    leading dimensions:
    -------------------
    
    FIXME: check if this is correct
    if A CblasRowMajor then the leading dimension switches it's place.
    E.g. A (M,K) CblasRowMajor matrix is a contiguous (M,lda) memory block
    
    The implementation decomposes the above expression as
    C = beta C
    C = alpha A B + C
    
    */
    
    int k,d,p;
    double *Ad, *Bd, *Cd;
    double *Ap, *Bp, *Cp;
    
    int pstrideA, pstrideB, pstrideC;
    int dstrideA, dstrideB, dstrideC;
    
    /* input checks */
    if( Order != 102){
        // TransA != 111 || TransB != 111 ||
        printf("The case TransA != 111 || TransB != 111 || Order != 101 has not been implemented yet!\n");
        return -1;
    }

    /* d > 0: higher order coefficients */
    
    if(TransA == 111) dstrideA = lda*K; 
    else if (TransA == 112) dstrideA = lda*M;
    
    if(TransB == 111)  dstrideB = ldb*N;
    else if(TransB == 112)  dstrideB = ldb*K;
    
    dstrideC = ldc*N;
    
    pstrideA = (D-1)*dstrideA; pstrideB = (D-1)*dstrideB; pstrideC = (D-1)*dstrideC;
    
    for(p = 0; p < P; ++p){
        Ap = A + p*pstrideA;
        Bp = B + p*pstrideB;
        Cp = C + p*pstrideC;
        
        for(d = D-1; 0 < d; --d){
            /* coefficient that is going to be updated */
            Cd = Cp + d * dstrideC;
            cblas_dgemm(Order, TransA, TransB, M, N, K, 0, A,
                 lda, B, ldb, beta, Cd, ldc);
                            
            /* compute C_d = beta C_d + alpha A_0 B_d */
            Bd = Bp + d * dstrideB;
            cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A,
                 lda, Bd, ldb, 1, Cd, ldc);
            
            /* compute sum_{k=1}^{d-1} x_k y_{d-k} */
            Ad = Ap + dstrideA;
            Bd = Bp + (d - 1) * dstrideB;
            for(k = 1; k < d; ++k){
                cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, Ad,
                 lda, Bd, ldb, 1, Cd, ldc);
                Ad += dstrideA;
                Bd -= dstrideB;
            }
            
            /* compute C_d = beta C_d + alpha A_d B_0 */
            Ad = Ap + d * dstrideA;
            cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, Ad,
                 lda, B, ldb, 1, Cd, ldc);
            
        }
    }
    
    /* d = 0: base point  C_0 = beta C_0 + alpha A_0 B_0 */
    cblas_dgemm(Order, TransA, TransB, M, N, K, 0, A,
         lda, B, ldb, 1, C, ldc);
    cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A,
                             lda, B, ldb, 1, C, ldc);
    return 0;
    
}


int utpm_dgesv(int P, int D, enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                  int N, int NRHS,
                  double *A, int lda, int *ipiv,
                  double *B, int ldb){
    /*
    Solves the linear system op(A) X = B in Taylor arithmetic,
    
    where op(A) = A or op(A) = A**T
    
    A (N,N) matrix.
    B (N,NRHS) matrix
    
    This is a modification of the API of dgesv.f: added possiblity enum CBLAS_TRANSPOSE TransA
    to the function argument list.Compare to http://www.netlib.org/lapack/double/dgesv.f.
    
    The solution is computed by:
    1) clapack_dgetrf (see http://www.netlib.org/lapack/double/dgetrf.f)
    2) clapack_dgetrs (see http://www.netlib.org/lapack/double/dgetrs.f)

    */
    
    int d,p;
    int dstrideA, dstrideB, pstrideA, pstrideB;
    double *Ad, *Bd;
    double *Ap, *Bp;
    
    int k;
    
    /* input checks */
    if(Order != 102){
        printf("Order != 101 has not been implemented yet!\n");
        return -1;
    }
    
    /* compute d = 0 */
    /* first A = P L U, i.e. LU with partial pivoting */
    
    clapack_dgetrf(Order, N, N, A, lda, ipiv);
    clapack_dgetrs(Order, TransA, N, NRHS, A, lda, ipiv, B, ldb);
    /* clapack_dgesv(Order, N, NRHS, A, lda, ipiv, B, ldb); */
    
    /* compute higher order coefficients d > 0 */
    dstrideA = lda*N; dstrideB = ldb*NRHS;
    pstrideA = dstrideA*(D-1); pstrideB = dstrideB*(D-1);
    
    for( p = 0; p < P; ++p){
        Ap = A + p*pstrideA;
        Bp = B + p*pstrideB;
        for( d = 1; d < D; ++d){
            /* compute B_d - \sum_{k=1}^d A_k B_{d-k} */
            for(k=1; k < d; ++k){
                Ad = Ap + k * dstrideA;
                Bd = Bp + (d-k) * dstrideB;
                /* FIXME: why the hell is now ldb = NRHS??? */
                cblas_dgemm(Order, TransA, CblasNoTrans, N, NRHS,
                     N, -1., Ad, lda, Bd, ldb, 1., Bp + d*dstrideB, ldb);
            }
            
            /* compute the last loop element, i.e. k = d */
            Ad = Ap + d*dstrideA;   Bd = Bp + d*dstrideB;
            
            /* FIXME: why the hell is now ldb = NRHS??? */
            cblas_dgemm(Order, TransA, CblasNoTrans, N, NRHS,
                 N, -1., Ad, lda, B, ldb, 1., Bd, ldb);
            
            /* compute solve(A_0,  B_d - \sum_{k=1}^d A_k B_{d-k}
               where A_0 is LU factorized already            */
            clapack_dgetrs(Order, TransA, N, NRHS, A, lda, ipiv, Bd, ldb);
        }
    }

    
    return 0;
} 
