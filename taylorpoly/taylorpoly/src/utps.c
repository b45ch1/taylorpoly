
int amul(int P, int D, double *x, double *y, double *z ){
    /* 
    assignment multiplication
    computes  z += mul(x,y) in Taylor arithmetic
    
    Latex::
        
        $u \cdot v $ & $\phi_d = phi_d  + \sum_{j=0}^d u_j  v_{d-j}$
        
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
        for(d = D-1; 0 < d; --d){
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

int mul(int P, int D, double *x, double *y, double *z ){
    /* 
    assignment multiplication
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

