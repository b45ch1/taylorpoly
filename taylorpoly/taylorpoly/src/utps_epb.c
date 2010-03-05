#include "utps.h"

int utps_pbe_add(int P, int D, double *x, double *y, double *z, double *zbar, double *xbar, double *ybar){
    int i;
    double *xbarq, *ybarq, *zbarq;
    
    xbarq = xbar;
    ybarq = ybar;
    zbarq = zbar;
    for( i = 0; i < (P*(D-1) + 1); ++i){
        *xbarq += *zbarq;
        *ybarq += *zbarq;
        ++xbarq; ++ybarq; ++zbarq;
    }
    return 0;
}

int utps_pbe_sub(int P, int D, double *x, double *y, double *z, double *zbar, double *xbar, double *ybar){
    int i;
    double *xbarq, *ybarq, *zbarq;
    
    xbarq = xbar;
    ybarq = ybar;
    zbarq = zbar;
    for( i = 0; i < (P*(D-1) + 1); ++i){
        *xbarq += *zbarq;
        *ybarq -= *zbarq;
        ++xbarq; ++ybarq; ++zbarq;
    }
    return 0;
}

int utps_pbe_mul(int P, int D, double *x, double *y, double *z, double *zbar, double *xbar, double *ybar){
    utps_amul(P, D, zbar, y, xbar );
    utps_amul(P, D, zbar, x, ybar );
    return 0;
}

