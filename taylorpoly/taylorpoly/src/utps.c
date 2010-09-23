#include <math.h>

int utps_add(int P, int D, double *x, double *y, double *z ){
    /* 
    computes  z = add(x,y) in Taylor arithmetic
    
    Latex::
        
       $ z_d = x_d + y_d$
        
    */
    
    int i;
    for(i = 0; i < 1 + (D-1)*P; ++i){
        (*z) = (*x) + (*y);
        ++z; ++x; ++y;
    }

    return 0;
}

int utps_sub(int P, int D, double *x, double *y, double *z ){
    /* 
    computes  z = sub(x,y) in Taylor arithmetic
    
    Latex::
        
       $ z_d = x_d - y_d$
        
    */
    int i;
    for(i = 0; i < 1 + (D-1)*P; ++i){
        (*z) = (*x) - (*y);
        ++z; ++x; ++y;
    }
    return 0;
}

int utps_mul(int P, int D, double *x, double *y, double *z ){
    /* 
    computes  z = mul(x,y) in Taylor arithmetic
    
    math::
        
        z_d = \sum_{j=0}^d x_j  y_{d-j} \quad \forall d = 0,...,D-1
        
    In the case when z = x or z = y (i.e. the pointers point at the same memory block,
    then x = mul(x,y) or y = mul(x,y) is computed.
        
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

int utps_div(int P, int D, double *x, double *y, double *z ){
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

inline int utps_amul(int P, int D, double *x, double *y, double *z ){
    /*
    computes  z += mul(x,y) in Taylor arithmetic
    */
    int k,d,p;
    double *zd, *xd, *yd;
    double *zp, *xp, *yp;
    double tmp;
    
    /* input checks */
    if(z == y || z == x) return -1;

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
            
            (*zd) += tmp;
        }
    }
    
    
    zd = z;
    yd = y;
    xd = x;
    
    /* d = 0: base point z_0 */
    (*zd) += (*xd) * (*yd);
    
    return 0;
}

int utps_sqrt(int P, int D, double *x, double *y){
    /* computes  y = sqrt(x) in Taylor arithmetic
    */
    
    int k,d,p;
    double *yd, *yd2;
    double *xp, *yp;
    double tmp;
    /* input checks */
    if(x == y) return -1;
    
    /* compute y_0 = log(x_0) */
    (*y) = sqrt(*x);
    
    /* d > 0: higher order coefficients */
    for(p = 0; p < P; ++p){
        xp = x + p*(D-1);
        yp = y + p*(D-1);
        
        for(d = 1; d < D; ++d){
            yd  = yp + 1;
            yd2 = yp + d - 1;
            tmp = *(xp+d);
            
            for(k = 1; k < d; ++k){
                tmp -= (*yd) * (*yd2);
                yd++;
                yd2--;
            }
            
            tmp /= (2*(*y));
            *(yp + d) = tmp;
        }
    }
    return 0;
}

int utps_log(int P, int D, double *x, double *y){
    /* computes  y = log(x) in Taylor arithmetic
    */
    
    int k,d,p;
    double *xd, *yd;
    double *xp, *yp;
    double tmp;
    /* input checks */
    if(x == y) return -1;
    
    /* compute y_0 = log(x_0) */
    (*y) = log(*x);
    
    /* d > 0: higher order coefficients */
    for(p = 0; p < P; ++p){
        xp = x + p*(D-1);
        yp = y + p*(D-1);
        
        for(d = 1; d < D; ++d){
            xd = xp + d;
            tmp = d*(*xd);
            xd--;
            
            yd = yp + 1;
            for(k = 1; k < d; ++k){
                tmp -= k*(*xd) * (*yd);
                xd--;
                yd++;
            }
            tmp /= (d*(*x));
            *(yp + d) = tmp;
        }
    }
    return 0;
}

int utps_exp(int P, int D, double *x, double *y){
    /* computes  y = exp(x) in Taylor arithmetic
    */
    
    int k,d,p;
    double *xd, *yd;
    double *xp, *yp;
    double tmp;
    /* input checks */
    if(x == y) return -1;
    
    /* compute y_0 = log(x_0) */
    (*y) = exp(*x);
    
    /* d > 0: higher order coefficients */
    for(p = 0; p < P; ++p){
        xp = x + p*(D-1);
        yp = y + p*(D-1);
        
        for(d = 1; d < D; ++d){
            xd = xp + 1;
            yd = yp + d - 1;
            tmp = 0;
            
            for(k = 1; k < d; ++k){
                tmp += k*(*yd) * (*xd);
                xd++;
                yd--;
            }
            
            tmp += d*(*y)*(*xd);
            
            tmp /= d;
            *(yp + d) = tmp;
        }
    }
    return 0;
}

int utps_pow(int P, int D, double *x, double r, double *y){
    /* computes  y = pow(x,r) in Taylor arithmetic
    */
    
    int k,d,p;
    double *xd, *yd;
    double *xp, *yp;
    double tmp;
    /* input checks */
    if(x == y) return -1;
    
    /* compute y_0 = log(x_0) */
    (*y) = pow(*x, r);
    
    /* d > 0: higher order coefficients */
    for(p = 0; p < P; ++p){
        xp = x + p*(D-1);
        yp = y + p*(D-1);
        
        for(d = 1; d < D; ++d){
            xd = xp + 1;
            yd = yp + d - 1;
            tmp = 0;
            
            /* compute r \sum_{k=1}^d y_{d-k} k x_k */
            for(k = 1; k < d; ++k){
                tmp += k*(*yd) * (*xd);
                xd++;
                yd--;
            }
            tmp += d*(*y)*(*xd);
            tmp *= r;
            
            /* compute \sum_{k=1}^{d-1} k* x_{d-k} y_{k} */
            xd = xp + d - 1;
            yd = yp + 1;
            for(k = 1; k < d; ++k){
                tmp -= k * (*xd) * (*yd);
                xd--;
                yd++;
            }
            
            tmp /= ( (*x) * d);
            *(yp + d) = tmp;
        }
    }
    return 0;
}

int utps_sin_cos(int P, int D, double *x, double *s, double *c){
    /* computes s = sin(x) and c = cos(x) in Taylor arithmetic
    */
    
    int k,d,p;
    double *xd, *sd, *cd;
    double *xp, *sp, *cp;
    double s_tmp, c_tmp;
    /* input checks */
    if(x == s || x == c || s == c) return -1;
    
    /* compute s_0 = sin(x_0) and c_0 = cos(x_0) */
    (*s) = sin(*x);
    (*c) = cos(*x);
    
    /* d > 0: higher order coefficients */
    for(p = 0; p < P; ++p){
        xp = x + p*(D-1);
        sp = s + p*(D-1);
        cp = c + p*(D-1);
        
        for(d = 1; d < D; ++d){
            xd = xp + 1;
            cd = cp + d - 1;
            sd = sp + d - 1;
            
            s_tmp = c_tmp = 0;
            
            for(k = 1; k < d; ++k){
                s_tmp += k * (*xd)*(*cd);
                c_tmp -= k * (*xd)*(*sd);
                xd++;
                sd--;
                cd--;
            }
            s_tmp += d*(*xd)*(*c);
            c_tmp -= d*(*xd)*(*s);
            
            s_tmp /= d;
            c_tmp /= d;
            
            *(sp + d) = s_tmp;
            *(cp + d) = c_tmp;
        }
    }
    return 0;
}

