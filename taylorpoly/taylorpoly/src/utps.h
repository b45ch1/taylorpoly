#ifndef UTPS_H
#define UTPS_H

int utps_add(int P, int D, double *x, double *y, double *z );
int utps_sub(int P, int D, double *x, double *y, double *z );
int utps_mul(int P, int D, double *x, double *y, double *z );
int utps_div(int P, int D, double *x, double *y, double *z );
inline int utps_amul(int P, int D, double *x, double *y, double *z );
int utps_sqrt(int P, int D, double *x, double *y);
int utps_log(int P, int D, double *x, double *y);
int utps_exp(int P, int D, double *x, double *y);
int utps_pow(int P, int D, double *x, double r, double *y);
int utps_sin_cos(int P, int D, double *x, double *s, double *c);

#endif
