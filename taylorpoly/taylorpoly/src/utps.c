
int amul(int D, int P, double *x, double *y, double *z ){
    /* 
    assignment multiplication
    computes  z += mul(x,y) in Taylor arithmetic
    */
    int k,d,p;
    double *zd, *xd, *yd;

    zd = z;
    yd = y;
    xd = x;
    
    /* d = 0: base point z_0 */
    (*zd) += (*xd) * (*yd);

    /* d > 0: higher order coefficients */
    for(d = 1; d < D; ++d){
        zd = z + 1 + P*(d-1);
        
        /* compute x_0 y_d */
        yd = y + (1 + P*(d-1));
        for(p = 0; p != P; ++p){
            (*zd) +=  x[0] * (*yd);
            yd++;
        }
        
        /* compute sum_{k=1}^{d-1} x_k y_{d-k} */
        xd = x + 1;
        for(k = 1; k < d; ++k){
            yd = y + (1 + P*(d-k-1));
            for(p = 0; p != P; ++p){
                (*zd) += (*xd) * (*yd);
                xd++;
                yd++;
            }
        }
        
        for(p = 0; p != P; ++p){
            (*zd) +=  (*xd) * y[0];
        }
        zd++;
    }
    
    return 0;
} 
