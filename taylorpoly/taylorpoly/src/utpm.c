#include <clapack.h>


int utpm_cauchy_product(int P, int D, int d, const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc){

    /* computes the convolution of two matrices, i.e.
     C[d] := alpha * sum ( op( A[:d+1] )*op( B[:d+1:-1] )) + beta * C[d],
     
     See http://www.netlib.org/blas/dgemm.f for documentation.
    */
    
    int p;
    int k, dstrideA, dstrideB, dstrideC, pstrideA, pstrideB, pstrideC;
    double *Ad, *Bd, *Cd;
    double *Ap, *Bp, *Cp;
    
    dstrideA = K*lda; dstrideB = N*ldb; dstrideC = N*ldc;
    pstrideA = dstrideA*(D-1); pstrideB = dstrideB*(D-1); pstrideC = dstrideC*(D-1);
    
    Ap = A + dstrideA; Bp = B + dstrideB; Cp = C + dstrideC;
    
    for(p=0; p < P; ++p){
        Bd = Bp + dstrideB * (d-1);
        Cd = Cp + dstrideC * (d-1);

        /* compute C_d = alpha A_0 B_d + beta C */
        cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A,
             lda, Bd, ldb, beta, Cd, ldc);
        
        for(k = 1; k < d; ++k){
        }
        
        /* compute C_d = alpha A_d B_0 + beta C_d */
        Ad = Ap + dstrideB * (d-1);
        cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, Ad,
             lda, B, ldb, beta, Cd, ldc);
        
    }
    
    return 0;
}

int utpm_daxpy(int P, int D, const int N, const double alpha, const double *x,
                 const int incx, double *y, const int incy){
    /*
    y = alpha * x + y
    
    See http://www.netlib.org/blas/daxpy.f for documentation.
    */
    
    /* input checks */
    if(x == y) return -1;
    
    cblas_daxpy(N + (D-1)*P*N, alpha, x, incx, y, incy);
    return 0;
}

double utpm_ddot(int P, int D, const int N, const double *x, const int incx,
    const double *y, const int incy){

    /*
    computes dot = x^T y
    
    See http://www.netlib.org/blas/ddot.f for documentation.
    
    */
        return 0;
    
    }


int utpm_dgesv(int P, int D, const enum CBLAS_ORDER Order,
                  const int N, const int NRHS,
                  double *A, const int lda, int *ipiv,
                  double *B, const int ldb){
    /*
    Solves the linear system A X = B in Taylor arithmetic.
    See the documentation of http://www.netlib.org/lapack/double/dgesv.f
    and http://www.netlib.org/lapack/double/dgetrs.f for details.
    The API is the modification of the clapack.h API.
    */
    
    int d,p;
    int k, dstrideA, dstrideB, pstrideA, pstrideB;
    double *Ad, *Bd;
    double *Ap, *Bp;
    
    /* compute d = 0 */
    clapack_dgesv(Order, N, NRHS, A, lda, ipiv, B, ldb);
    
    /* compute higher order coefficients d > 0 */


    
    return 0;
}

