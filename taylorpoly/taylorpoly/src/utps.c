
int amul(int P, int D, double *x, double *y, double *z ){
    /* 
    assignment multiplication
    computes  z += mul(x,y) in Taylor arithmetic
    */
    int k,d,p;
    double *zd, *xd, *yd;
    double *zp, *xp, *yp;

    zd = z;
    yd = y;
    xd = x;
    
    /* d = 0: base point z_0 */
    (*zd) += (*xd) * (*yd);

    for(p = 0; p < P; ++p){
        xp = x + p*(D-1);
        yp = y + p*(D-1);        
        zp = z + p*(D-1);
        
        /* d > 0: higher order coefficients */
        for(d = 1; d < D; ++d){
            zd = zp + d;
            
            /* compute x_0 y_d */
            yd = yp + d;
            (*zd) +=  x[0] * (*yd);
            
            /* compute sum_{k=1}^{d-1} x_k y_{d-k} */
            xd = xp + 1;
            yd = yp + d - 1;
            for(k = 1; k < d; ++k){
                (*zd) += (*xd) * (*yd);
                xd++;
                yd--;
            }
            
            /* compute x_d y_0 */
            xd = xp + d;
            (*zd) +=  (*xd) * y[0];        
        }
    }
    return 0;
} 
