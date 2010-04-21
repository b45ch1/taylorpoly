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

inline int get_leadim_and_cblas_transpose(int M, int N, int *strides, int *leadim, int *trans){
    /* 
    INPUTS:
    
        M              number of rows of the matrix
        N              number of cols of the matrix
        strides        int array of size 3, used as in numpy
    
    OUTPUTS:
    
        leadim is the leading dimension as blas and lapack require it
        trans indicates if the strides specify an actually transposed matrix
    */
    int is_transposed;
    is_transposed = (strides[1] < strides[0]);
    
    // printf("strides[1] = %d, strides[0] = %d, is_transposed = %d\n",strides[1],strides[0], is_transposed);
    
    if (is_transposed) *trans = CblasTrans;
    else *trans = CblasNoTrans;
    
    if (!is_transposed) *leadim = strides[1]/sizeof(double);
    else *leadim = N;
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
    dstrideA = Astrides[2]/sizeof(double);
    dstrideB = Bstrides[2]/sizeof(double);
    dstrideC = Cstrides[2]/sizeof(double);
    
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


/*
int utpm_qr(int P, int D, int M, int N, double *A, int ldA, double *Q, int ldQ, double *R, int ldR){
    
    // void lapack_dgeqrf(const int m, const int n, double * a, const int lda, double * tau, double * work, const int lwork, int * info );

    
}
*/




