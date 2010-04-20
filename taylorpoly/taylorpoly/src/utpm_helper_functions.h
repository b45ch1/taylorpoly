#ifndef UTPM_HELPER_FUNCTIONS_H
#define UTPM_HELPER_FUNCTIONS_H

inline int imul(int M, int N, double *y, int ldy, double *x, int ldx); 
inline int idiv(int M, int N, double *y, int ldy, double *x, int ldx);
inline int amul(int M, int N, double *x, int ldx, double *y, int ldy, double *z, int ldz);


#endif
