inline int imul(int M, int N, double *y, int ldy, double *x, int ldx){
    /* computes y *= x, where y and x are (M,N) arrays stored in column major format
    with leading dimensions ldx and ldy.
    */
    
    int n, m;
    
    for(n = 0; n < N; ++n){
        for(m = 0; m < M; ++m){
            (*y) *=  (*x);
            ++y; ++x;
        }
        y += (ldy - M);
        x += (ldx - M);
    }

    return 0;
}

inline int amul(int M, int N, double *x, int ldx, double *y, int ldy, double *z, int ldz){
    /* computes z = z + x * y, where y and x are (M,N) arrays stored in column major format
    with leading dimensions ldx and ldy.
    */
    
    int n, m;
    
    for(n = 0; n < N; ++n){
        for(m = 0; m < M; ++m){
            (*z) += (*y) * (*x);
            ++x; ++y; ++z;
        }
        x += (ldx - M);
        y += (ldy - M);
        z += (ldz - M);
    }

    return 0;
}




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





