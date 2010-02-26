int add(int P, int D, double *x, double *y, double *z ){
    /* 
    computes  z = add(x,y) in Taylor arithmetic
    
    Latex::
        
       $ z_d = x_d + y_d$
        
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

int sub(int P, int D, double *x, double *y, double *z ){
    /* 
    computes  z = sub(x,y) in Taylor arithmetic
    
    Latex::
        
       $ z_d = x_d - y_d$
        
    */
    int d,p;
    double *zd, *xd, *yd;
    double *zp, *xp, *yp;
    
    zd = z;
    yd = y;
    xd = x;
    
    /* d = 0: base point z_0 */
    (*zd) = (*xd) - (*yd);

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
            (*zd) = (*xd) - (*yd);
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


int div(int P, int D, double *x, double *y, double *z ){
    /* 
    computes  z = div(x,y) in Taylor arithmetic
    
    Latex::
        
        $z = x / y $ &
        $\z_d = \frac{1}{y_0} \left[ x_d - \sum_{k=0}^{d-1} \z_j y_{d-k} \right]$
        
    */
    int k,d,p;
    double *zd, *zd2, *xd, *yd;
    double *zp, *xp, *yp;
    
    /* input checks */
    if(z == y) return -1;
    
    /* d = 0: compute z_0 = x_0 / y_0 */
    (*z) = (*x)/(*y);

    /* d > 0: higher order coefficients */
    for(p = 0; p < P; ++p){
        xp = x + p*(D-1);
        yp = y + p*(D-1);
        zp = z + p*(D-1);
        for(d = 1; d < D; ++d){
            zd = zp + d;
            xd = xp + d;
            
            /* set z_d = x_d */
            (*zd) = (*xd);
            
            yd = yp + d;
            
            /* compute x_d - z_0 y_{d} */
            (*zd) -= (*z) * (*yd);
            yd--;
            
            /* compute x_d - \sum_{k=1}^{d-1} z_k y_{d-k} */
            zd2 = zp+1;
            for(k = 1; k < d; ++k){
                (*zd) -= (*zd2) * (*yd);
                zd2++;
                yd--;
            }
            
            /* compute 1./y_0 (x_d - \sum_{k=0}^{d-1} z_k y_{d-k}) */
            (*zd) /= (*y);
        }

    }
    
    return 0;
}

