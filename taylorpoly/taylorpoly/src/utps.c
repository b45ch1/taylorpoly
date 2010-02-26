int add(int P, int D, double *x, double *y, double *z ){
    /* 
    computes  z = add(x,y) in Taylor arithmetic
    
    Latex::
        
       $ z_d = x_d y_d$
        
    */
    int d,p;
    double *zd, *xd, *yd;
    double *zp, *xp, *yp;
    
    zd = z;
    yd = y;
    xd = x;
    
    /* d = 0: base point z_0 */
    (*zd) = (*xd) + (*yd);

    /* d > 0: higher order coefficients */
    for(p = 0; p < P; ++p){
        xp = x + p*(D-1);
        yp = y + p*(D-1);
        zp = z + p*(D-1);
        
        /* compute x_d y_d for 0<d<D */
        for(d = 1; d < D; ++d){
            zd = zp + d;
            xd = xp + d;
            yd = yp + d;
            (*zd) = (*xd) + (*yd);
        }
    }
    return 0;
}



int mul(int P, int D, double *x, double *y, double *z ){
    /* 
    computes  z = mul(x,y) in Taylor arithmetic
    
    Latex::
        
        $u \cdot v $ & $\phi_d = \sum_{j=0}^d u_j  v_{d-j}$
        
    */
    int k,d,p;
    double *zd, *xd, *yd;
    double *zp, *xp, *yp;
    double tmp;

    /* d > 0: higher order coefficients */
    for(p = 0; p < P; ++p){
        xp = x + p*(D-1);
        yp = y + p*(D-1);
        zp = z + p*(D-1);
        
        for(d = D-1; 0 < d; --d){
            zd = zp + d;
            
            tmp = 0;
            
            /* compute x_0 y_d */
            yd = yp + d;
            tmp +=  x[0] * (*yd);
            
            /* compute sum_{k=1}^{d-1} x_k y_{d-k} */
            xd = xp + 1;
            yd = yp + d - 1;
            for(k = 1; k < d; ++k){
                tmp += (*xd) * (*yd);
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
    
    return 0;
}

