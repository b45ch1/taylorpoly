#include <clapack.h>


int utpm_cauchy_product(int P, int D, int p, int d, int dA, int dB, const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc){

    /* computes the convolution of two matrices, i.e.
     C := alpha * sum ( op( A[dA:dA+d+1] )*op( B[dB:dB+d+1:-1] )) + beta * C,
     
     for one direction p.
     
     See http://www.netlib.org/blas/dgemm.f for documentation.
    */
    
    int k, dstrideA, dstrideB;
    const double *Ad, *Bd;
    
    dstrideA = K*lda; dstrideB = N*ldb;
    
    if(dA > 0 && dB == 0) {
        Ad = A + dstrideA;
        Bd = B + dstrideB + p*dstrideB*(dB + d - 1);
        
        for(k = dA + 1; k < d+dA; ++k){
            cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, Ad,
             lda, B, ldb, beta, C, ldc);
                                                                      
            Ad += dstrideA; Bd -= dstrideB;
        }
        
        /* compute C_d = alpha A_d B_0 + beta C_d */
        cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, Ad,
             lda, B, ldb, beta, C, ldc);
            
        return 0;
    }
    else{
        return -1;
    }
    
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
    int dstrideA, dstrideB, pstrideA, pstrideB;
    double *Bd;
    double *Bp;
    
    /* compute d = 0 */
    clapack_dgesv(Order, N, NRHS, A, lda, ipiv, B, ldb);
    
    /* compute higher order coefficients d > 0 */
    
    /*
        # d = 0:  base point
        for p in range(P):
            y_data[0,p,...] = numpy.linalg.solve(A_data[0,p,...], x_data[0,p,...])

        # d = 1,...,D-1
        tmp = numpy.zeros((M,K),dtype=float)
        for d in range(1, D):
            for p in range(P):
                tmp[:,:] = x_data[d,p,:,:]
                for k in range(1,d+1):
                    tmp[:,:] -= numpy.dot(A_data[k,p,:,:],y_data[d-k,p,:,:])
                y_data[d,p,:,:] = numpy.linalg.solve(A_data[0,p,:,:],tmp)
    
    */
    
    dstrideA = N*lda; dstrideB = NRHS*ldb;
    pstrideA = dstrideA*(D-1); pstrideB = dstrideB*(D-1);
    
    Bp = B + dstrideB;
    for( p = 0; p < P; ++p){
        Bd = Bp;
        for( d = 1; d < D; ++d){
            /* compute B_d - \sum_{k=1}^d A_k B_{d-k} */
            utpm_cauchy_product(P, D, p, d, 1, 0, Order, CblasNoTrans, CblasNoTrans,
                              N, NRHS, N, -1., A, lda, B, ldb, 1., Bd, ldb);
            
            /* compute solve(A_0,  B_d - \sum_{k=1}^d A_k B_{d-k}
               where A_0 is LU factorized already            */
            clapack_dgetrs(Order, CblasNoTrans, N, NRHS, A, lda, ipiv, Bd, ldb);
            
            Bd += dstrideB;
        }
        Bp += pstrideB;
    }

    
    return 0;
}

