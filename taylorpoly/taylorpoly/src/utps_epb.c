#include "utps.h"

int utps_epb_add(int P, int D, double *x, double *y, double *z, double *zbar, double *xbar, double *ybar){
    utps_add(P, D, xbar, zbar, xbar);
    utps_add(P, D, ybar, zbar, ybar);
    return 0;
}

int utps_epb_sub(int P, int D, double *x, double *y, double *z, double *zbar, double *xbar, double *ybar){
    utps_add(P, D, xbar, zbar, xbar);
    utps_sub(P, D, ybar, zbar, ybar);
    return 0;
}

int utps_epb_mul(int P, int D, double *x, double *y, double *z, double *zbar, double *xbar, double *ybar){
    utps_amul(P, D, zbar, y, xbar );
    utps_amul(P, D, zbar, x, ybar );
    return 0;
}

int utps_epb_div(int P, int D, double *x, double *y, double *z, double *zbar, double *xbar, double *ybar){
    utps_amul(P, D, zbar, z, xbar );
    utps_mul(P, D, z, z, z);
    utps_amul(P, D, zbar, z, ybar );
    return 0;
}

