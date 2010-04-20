/* Algorithms for Univariate Taylor Polynomial over Matrices.
A matrix is a 2D array.
*/

#include "utpm_helper_functions.h"
#include "utpm_blas_lapack.h"

inline int utpm_imul(int P, int D, int M, int N, double *y, int ldy, double *x, int ldx){
    
    int k,d,p;
    double *xd, *yd, *zd;
    double *xp, *yp, *zp;
    
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


int utpm_qr(int P, int D, int M, int N, double *A, int ldA, double *Q, int ldQ, double *R, int ldR){
    
    // void lapack_dgeqrf(const int m, const int n, double * a, const int lda, double * tau, double * work, const int lwork, int * info );

}





