#include <cblas.h>
#include <clapack.h>
#include <stdio.h>




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

int daxmy(int P, int D, int N, double alpha, double *x, int incx, double *y, int incy){
    /* y = alpha * x * y, where x,y double arrays of lenght N */
    int p,d,n;
    double *yp, *xp, *yd, *xd;
    
    for(p = 0; p < P; ++p){
        for(n = 0; n < N; n += incx;){
            *y = alpha * (*x) * 
        }
    }

}

int utpm_daxmy(int P, int D, int N, double alpha, double *x,  int incx, double *y, int incy){
    /*
    y = alpha * x * y
    */
    
    int k,p,d,n;
    double *zd, *xd, *yd;
    double *zp, *xp, *yp;
    int pstride;
    
    /* set pointer z to be equal to y and compute z = alpha * x * y */
    z = y;
    
    /* set the strides */
    pstride = (D-1)*N;

    /* d > 0: higher order coefficients */
    for(p = 0; p < P; ++p){
        xp = x + p*pstride;
        yp = y + p*pstride;
        zp = z + p*pstride;
        
        for(d = D-1; 0 < d; --d){
            zd = zp + d*N;
            
            /* compute x_0 y_d */
            yd = yp + d;
            cblas_daxpy(N, alpha*x[0], yd, incy, zd, incy);
            
            /* compute sum_{k=1}^{d-1} x_k y_{d-k} */
            xd = xp + 1;
            yd = yp + d - 1;
            for(k = 1; k < d; ++k){
                cblas_daxpy(N, 
                zd += (*xd) * (*yd);
                xd++;
                yd--;
            }
            
            /* compute x_d y_0 */
            xd = xp + d;
            tmp +=  (*xd) * y[0];
            
            (*zd) = tmp;
        }
    }
    
    
    zd = z;
    yd = y;
    xd = x;
    
    /* d = 0: base point z_0 */
    (*zd) = (*xd) * (*yd);
    
    

}

int utpm_dgemm(int P, int D, enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
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
    if(TransA != 111 || TransB != 111 || Order != 102){
        printf("The case TransA != 111 || TransB != 111 || Order != 101 has not been implemented yet!\n");
        return -1;
    }

    /* d > 0: higher order coefficients */
    dstrideA = lda*K; dstrideB = ldb*N; dstrideC = ldc*N;
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

// int utpm_cauchy_product(int P, int D, int p, int d, int dA, int dB, enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
//                  enum CBLAS_TRANSPOSE TransB, int M, int N,
//                  int K, double alpha, double *A,
//                  int lda, double *B, int ldb,
//                  double beta, double *C, int ldc){

//     /* computes the convolution of two matrices, i.e.
//      C := alpha * sum ( op( A[dA:dA+d] )*op( B[dB:dB+1:-1] )) + beta * C,
     
//      for one direction p.
     
//      See http://www.netlib.org/blas/dgemm.f for documentation.
//     */
    
//     int k, dstrideA, dstrideB;
//     double *Ad, *Bd;
    
//     dstrideA = K*lda; dstrideB = N*ldb;
    
//     if(dA > 0 && dB == 0) {
//         Ad = A + dstrideA;
//         Bd = B + dstrideB + p*dstrideB*(dB + d - 1);
        
//         for(k = dA; k < d - 1 +dA; ++k){
//             cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, Ad,
//              lda, Bd, ldb, beta, C, ldc);
                                                                      
//             Ad += dstrideA; Bd -= dstrideB;
//         }
        
//         /* compute C_d = alpha A_d B_0 + beta C_d */
//         cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, Ad,
//              lda, B, ldb, beta, C, ldc);
            
//         return 0;
//     }
//     else{
//         return -1;
//     }
// }

