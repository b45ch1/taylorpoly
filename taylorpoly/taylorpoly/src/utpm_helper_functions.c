inline int imul(int M, int N, double *y, int ldy, double *x, int ldx){
    /* computes y *= x, where y and x are (M,N) arrays stored in column major format
    with leading dimensions ldx and ldy.
    */
    
    int n, m;
    
    for(n = 0; n < N; ++n){
        for(m = 0; m < M; ++m){
            (*y) *=  (*x);
            ++y; ++x;
        }
        y += (ldy - M);
        x += (ldx - M);
    }

    return 0;
}

inline int idiv(int M, int N, double *y, int ldy, double *x, int ldx){
    /* computes y /= x, where y and x are (M,N) arrays stored in column major format
    with leading dimensions ldx and ldy.
    */
    
    int n, m;
    
    for(n = 0; n < N; ++n){
        for(m = 0; m < M; ++m){
            (*y) /=  (*x);
            ++y; ++x;
        }
        y += (ldy - M);
        x += (ldx - M);
    }

    return 0;
}

inline int amul(int M, int N, double *x, int ldx, double *y, int ldy, double *z, int ldz){
    /* computes z = z + x * y, where y and x are (M,N) arrays stored in column major format
    with leading dimensions ldx and ldy.
    */
    
    int n, m;
    
    for(n = 0; n < N; ++n){
        for(m = 0; m < M; ++m){
            (*z) += (*y) * (*x);
            ++x; ++y; ++z;
        }
        x += (ldx - M);
        y += (ldy - M);
        z += (ldz - M);
    }

    return 0;
} 
