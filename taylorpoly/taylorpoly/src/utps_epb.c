#include "utps.h"

int utps_epb_add(int Q, int P, int D, double *x, double *y, double *z, double *zbar, double *xbar, double *ybar){
    int q, size;
    size = 1 + (D-1)*P;
    for(q = 0; q < Q; ++q){
        utps_add(P, D, xbar, zbar, xbar);
        utps_add(P, D, ybar, zbar, ybar);
        xbar += size;
        ybar += size;
        zbar += size;
    }

    return 0;
}

int utps_epb_sub(int Q, int P, int D, double *x, double *y, double *z, double *zbar, double *xbar, double *ybar){
    int q, size;
    size = 1 + (D-1)*P;
    for(q = 0; q < Q; ++q){
        utps_add(P, D, xbar, zbar, xbar);
        utps_sub(P, D, ybar, zbar, ybar);
        xbar += size;
        ybar += size;
        zbar += size;
    }
    
    return 0;
}

int utps_epb_mul(int Q, int P, int D, double *x, double *y, double *z, double *zbar, double *xbar, double *ybar){
    int q, size;
    size = 1 + (D-1)*P;
    for(q = 0; q < Q; ++q){
        utps_amul(P, D, zbar, y, xbar );
        utps_amul(P, D, zbar, x, ybar );
        xbar += size;
        ybar += size;
        zbar += size;
    }
    
    return 0;
}

int utps_epb_div(int Q, int P, int D, double *x, double *y, double *z, double *zbar, double *xbar, double *ybar){
    int err_code;
    int q, size;
    err_code = 0;
    size = 1 + (D-1)*P;
    for(q = 0; q < Q; ++q){
        err_code += utps_div(P, D, zbar, y, zbar );
        err_code += utps_add(P, D, xbar, zbar, xbar);
        err_code += utps_mul(P, D, zbar, z, zbar );
        err_code += utps_sub(P, D, ybar, zbar, ybar );
        xbar += size;
        ybar += size;
        zbar += size;
    }
    
    return err_code;
}

