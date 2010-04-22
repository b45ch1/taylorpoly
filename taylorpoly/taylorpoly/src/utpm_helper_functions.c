#include <cblas.h>

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

inline int get_leadim_and_cblas_transpose(int M, int N, int *strides, int *leadim, int *trans){
    /* 
    INPUTS:
    
        M              number of rows of the matrix
        N              number of cols of the matrix
        strides        int array of size 3, used as in numpy, layout (8*D, strides of matrix)
    
    OUTPUTS:
    
        leadim is the leading dimension as blas and lapack require it
        trans indicates if the strides specify an actually transposed matrix
    */
    int is_transposed;
    is_transposed = (strides[2] < strides[1]);
    
    // printf("strides[2] = %d, strides[1] = %d, is_transposed = %d\n",strides[2],strides[1], is_transposed);
    
    if (is_transposed) *trans = CblasTrans;
    else *trans = CblasNoTrans;
    
    if (!is_transposed) *leadim = strides[2]/sizeof(double);
    else *leadim = N;
    return 0;
}

inline int l_times_u(int N, double alpha, double *A, int lda, double *B, int ldb, double *C, int ldc){
    /* 
    computes A := A + alpha * L * U,
    where L lower triangular matrix and U upper triangular matrix
    
    (inefficient) helper function that implements functionality missing in BLAS
    
    */
    int m,n,i;
    int itmp;
    double dtmp;
    double *Ai, *Bi, *Ci;
    
    Ai = A;
    for(m = 0; m < N; ++m){
        for(n = 0; n < N; ++n){
            itmp = (m <= n ? m : n);
            dtmp = 0;
            Bi = B + m + n*ldb;
            Ci = C + n*ldc;
            for(i = 0; i <= itmp; ++i){
                dtmp += alpha * (*Bi) * (*Ci);
                Bi += ldb;
                ++Ci;
            }
            (*Ai) += dtmp;
            ++Ai;
        }
        Ai += (lda - N);
    }
    return 0;
}


