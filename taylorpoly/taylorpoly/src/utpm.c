#include <clapack.h>

int utpm_dgesv(int P, int D, const enum CBLAS_ORDER Order,
                  const int N, const int NRHS,
                  double *A, const int lda, int *ipiv,
                  double *B, const int ldb){
    /*
    Solves the linear system A X = B in Taylor arithmetic.
    See the documentation of http://www.netlib.org/lapack/double/dgesv.f for details.
    The API is the modification of the clapack.h API.
    */
    
    int info;
    
    /* compute d = 0 */
    info = clapack_dgesv(Order, N, NRHS, A, lda, ipiv, B, ldb);
    
    return info;
}

