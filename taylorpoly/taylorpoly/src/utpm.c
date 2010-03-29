#include <clapack.h>


int utpm_cauchy_product(int P, int D, int p, int d, int dA, int dB, enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, int M, int N,
                 int K, double alpha, double *A,
                 int lda, double *B, int ldb,
                 double beta, double *C, int ldc){

    /* computes the convolution of two matrices, i.e.
     C := alpha * sum ( op( A[dA:dA+d] )*op( B[dB:dB+1:-1] )) + beta * C,
     
     for one direction p.
     
     See http://www.netlib.org/blas/dgemm.f for documentation.
    */
    
    int k, dstrideA, dstrideB;
    double *Ad, *Bd;
    
    dstrideA = K*lda; dstrideB = N*ldb;
    
    if(dA > 0 && dB == 0) {
        Ad = A + dstrideA;
        Bd = B + dstrideB + p*dstrideB*(dB + d - 1);
        
        for(k = dA; k < d - 1 +dA; ++k){
            cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, Ad,
             lda, Bd, ldb, beta, C, ldc);
                                                                      
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

int utpm_dscal(int P, int D, int N, double alpha, double *X, int incX){
    cblas_dscal(N + (D-1)*P*N, alpha, X, incX);
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

double utpm_ddot(int P, int D, int N, double *x, int incx,
    double *y, int incy){

    /*
    computes dot = x^T y
    
    See http://www.netlib.org/blas/ddot.f for documentation.
    
    */
        return 0;
    
}


int utpm_dgemm(int P, int D, enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, int M, int N,
                 int K, double alpha, double *A,
                 int lda, double *B, int ldb,
                 double beta, double *C, int ldc){

    /* See http://www.netlib.org/blas/dgemm.f for documentation
    
    C = alpha A B + beta C
    
    The implementation decomposes the above expression as
    C = beta C
    C = alpha A B + C
    
    */
    
    int k,d,p;
    double *Ad, *Bd, *Cd;
    double *Ap, *Bp, *Cp;
    
    int pstrideA, pstrideB, pstrideC;
    int dstrideA, dstrideB, dstrideC;

    /* d > 0: higher order coefficients */
    dstrideA = K*lda; dstrideB = N*ldb; dstrideC = N*ldb;
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


int utpm_dgesv(int P, int D, enum CBLAS_ORDER Order,
                  int N, int NRHS,
                  double *A, int lda, int *ipiv,
                  double *B, int ldb){
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

