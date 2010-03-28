#include <clapack.h>


int utpm_daxpy(int P, int D, const int N, const double alpha, const double *x,
                 const int incx, double *y, const int incy){
    /*
    y = alpha * x + y
    
    See http://www.netlib.org/blas/daxpy.f for documentation.
    */
    
    /* input checks */
    if(x == y) return -1;
    
    cblas_daxpy(N + (D-1) * P * N, alpha, x, incx, y, incy);
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
    double *Ad, *Bd, *Xd;
    double *Ap, *Bp, *Xp;
    int info;
    
    /* compute d = 0 */
    info = clapack_dgesv(Order, N, NRHS, A, lda, ipiv, B, ldb);
    
    /* compute higher order coefficients d > 0 */
    for(p=0; p < P; ++p){
        
        for(d=1; d < D; ++d){
            int clapack_dgetrs(Order, CblasNoTrans, N, NRHS, A, lda, ipiv, B, ldb);
        }
    }

    
    return info;
}

