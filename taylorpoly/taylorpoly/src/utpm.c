/* Algorithms for Univariate Taylor Polynomial over Matrices.
A matrix is a 2D array.
*/
#include <cblas.h>

#include "utpm_helper_functions.h"
#include "utpm_blas_lapack.h"



inline int utpm_imul(int P, int D, int M, int N, double *y, int ldy, double *x, int ldx){
    
    int k,d,p;
    double *xd, *yd, *zd;
    double *xp, *yp;
    
    int pstridex, pstridey;
    int dstridex, dstridey;
    
    dstridex = ldx*N;
    dstridey = ldy*N;
    
    pstridex = (D-1)*dstridex;
    pstridey = (D-1)*dstridey;

    /* d > 0: higher order coefficients */
    for(p = 0; p < P; ++p){
        xp = x + p*pstridex;
        yp = y + p*pstridey;
        
        for(d = D-1; 0 < d; --d){
            
            /* compute y_d += x_0 y_d */
            xd = x;
            yd = yp + d*dstridey;
            zd = yd;
            imul(M, N, yd, ldy, xd, ldx);
            
            /* compute y_d += sum_{k=1}^{d-1} x_k y_{d-k} */
            xd = xp + dstridex;
            yd = yp + (d-1)*dstridey;
            
            for(k = 1; k < d; ++k){
                amul(M, N, xd, ldx, yd, ldy, zd, ldy);
                ++xd;
                yd -= (2*dstridey-1);
            }
            
            /* compute y_d += x_d y_0 */
            yd = y;
            xd = xp + d*dstridex;
            amul(M, N, xd, ldx, yd, ldy, zd, ldy);
        }
    }
    
    yd = y;
    xd = x;
    /* d = 0: base point z_0 */
    imul(M, N, yd, ldy, xd, ldx);
    
    return 0;
}

int utpm_dot(int P, int D, int M, int N, int K, double alpha, double *A,
                 int *Astrides, double *B, int *Bstrides,
                 double beta, double *C, int *Cstrides){

    /* C = alpha A B + beta C
    
    dimensions:
    -----------
    A is a (M,K) matrix
    B is a (K,N) matrix
    C is a (M,N) matrix
    
    Astrides, Bstrides and Cstrides are int arrays of length 3.
    E.g. Astrides[0] = 24 means that 24/8 = 3 Bytes have to be omitted to get from A[i,j,k] to A[i+i,j,k].
    
    The implementation decomposes the above expression as
    C = beta C
    C = alpha A B + C
    
    */
    
    // printf("Astrides[0] = %d, Astrides[1] = %d, Astrides[2] = %d\n",Astrides[0], Astrides[1], Astrides[2]);
    
    int k,d,p;
    double *Ad, *Bd, *Cd;
    double *Ap, *Bp, *Cp;
    
    int lda, ldb, ldc, TransA, TransB, TransC;
    int Order;
    
    int pstrideA, pstrideB, pstrideC;
    int dstrideA, dstrideB, dstrideC;
    
    /* input checks */

    /* d > 0: higher order coefficients */
    Order = CblasColMajor;
    dstrideA = Astrides[0]/sizeof(double);
    dstrideB = Bstrides[0]/sizeof(double);
    dstrideC = Cstrides[0]/sizeof(double);
    
    get_leadim_and_cblas_transpose(M, K, Astrides, &lda, &TransA);
    get_leadim_and_cblas_transpose(K, N, Bstrides, &ldb, &TransB);
    get_leadim_and_cblas_transpose(M, N, Cstrides, &ldc, &TransC);
    
    // printf("lda = %d, ldb = %d, ldc = %d\n",lda,ldb,ldc);
    
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


int utpm_solve(int P, int D, int N, int NRHS, int *ipiv, double *A, int *Astrides,
                  double *B, int *Bstrides){
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
    
    int d,p,k;
    int dstrideA, dstrideB, pstrideA, pstrideB;
    double *Ad, *Bd;
    double *Ap, *Bp;
    
    int lda, ldb, TransA, TransB;
    int Order;
    
    /* prepare stuff for the lapack call */
    Order = CblasColMajor;
    get_leadim_and_cblas_transpose(N, N, Astrides, &lda, &TransA);
    get_leadim_and_cblas_transpose(N, NRHS, Bstrides, &ldb, &TransB);
    
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
            
            cblas_dgemm(Order, TransA, CblasNoTrans, N, NRHS,
                 N, -1., Ad, lda, B, ldb, 1., Bd, ldb);
            
            /* compute solve(A_0,  B_d - \sum_{k=1}^d A_k B_{d-k}
               where A_0 is LU factorized already            */
            clapack_dgetrs(Order, TransA, N, NRHS, A, lda, ipiv, Bd, ldb);
        }
    }
    return 0;
}

int utpm_lu(int P, int D, int N, int *ipiv, double *A, int *Astrides, double *work){
  
    int d,p,k;
    int m,n,j;
    int dstrideA, pstrideA;
    double *Ad, *Ld, *Ud;
    double *Ap;
    int lda, TransA;
    int Order;
    
    int itmp;
    double dtmp;
    
    /* prepare stuff for the lapack call */
    Order = CblasColMajor;
    get_leadim_and_cblas_transpose(N, N, Astrides, &lda, &TransA);
    dstrideA = Astrides[0]/sizeof(double);
    pstrideA = (D-1)*dstrideA;
    
    /* compute d = 0 */
    /* first A = P L U, i.e. LU with partial pivoting */
    clapack_dgetrf(Order, N, N, A, lda, ipiv);
    
    /* compute higher order coefficients d > 0 */

    for( p = 0; p < P; ++p){
        Ap = A + p*pstrideA;
        for( d = 1; d < D; ++d){
            /* compute Delta F = A_d - \sum_{k=1}^d A_k B_{d-k} */
            Ad = Ap + d * dstrideA;
            for(k=1; k < d; ++k){
                Ld = Ap + k * dstrideA;
                Ud = Ap + (d-k) * dstrideA;
                l_times_u(N, -1., Ad, lda, Ld, lda, Ud, lda);
            }
            // lapack_dtrtrs(lapack_lower, lapack_no_trans, lapack_unit_diag, N, N, const double * a, const int lda, double * b, const int ldb, int * info );
        }
    }
    
    
    return 0;
}


