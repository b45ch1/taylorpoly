#ifndef UTPS_EPB_H
#define UTPS_EPB_H

int utps_epb_add(int P, int D, double *x, double *y, double *z, double *zbar, double *xbar, double *ybar);
int utps_epb_sub(int P, int D, double *x, double *y, double *z, double *zbar, double *xbar, double *ybar);
int utps_epb_mul(int P, int D, double *x, double *y, double *z, double *zbar, double *xbar, double *ybar);
int utps_epb_div(int P, int D, double *x, double *y, double *z, double *zbar, double *xbar, double *ybar);

#endif
