inline int imul(int M, int N, double *y, int ldy, double *x, int ldx){
    /* computes y *= x, where y and x are (M,N) arrays stored in column major format
    with leading dimensions ldx and ldy.
    */
    
    int n, m;
    
    for(n = 0; n < N;){
        for(m = 0; m < M;){
            (*y) +=  (*x) * (*y);
            ++y; ++x;
        }
        y += (ldy - M);
        x += (ldx - M);
    }

    return 0;
}


int utpm_imul(int P, int D, int M, int N, double *y, int ldy, double *x, int ldx){
    
    int k,d,p;
    double *xd, *yd;
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
            
            /* compute y_d = x_0 y_d */
            yd = yp + d*dstridey;
            xd = x;

            
            /* compute y_d = sum_{k=1}^{d-1} x_k y_{d-k} */
            xd = xp + dstridex;
            yd = yp + (d-1)*dstridey;
            
            for(k = 1; k < d; ++k){
                imul(M, N, xd, ldx, yd, ldy);
                ++xd;
                yd -= (2*dstridey-1);
            }
            
            /* compute y_d = x_d y_0 */
            yd = y;
            xd = xp + d*dstridex;
            imul(M, N, xd, ldx, yd, ldy);
        }
    }
    
    yd = y;
    xd = x;
    /* d = 0: base point z_0 */
    imul(M, N, xd, ldx, yd, ldy);
    
    return 0;
    
}





