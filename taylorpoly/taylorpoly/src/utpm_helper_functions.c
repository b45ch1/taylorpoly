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
        strides        int array of size 3, used as in numpy, layout (sizeof(double)*D, strides of matrix)
    
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

inline int l_times_u(int N, double alpha, double *A, int lda, double *L, int ldl, double *U, int ldu){
    /* 
    computes A := A + alpha * L * U,
    where L unit lower triangular matrix and U upper triangular matrix
    
    (inefficient) helper function that implements functionality missing in BLAS
    
    */
    int m,n,i;
    int itmp;
    double dtmp;
    double *Ai, *Li, *Ui;
    
    Ai = A;
    for(n = 0; n < N; ++n){
        for(m = 0; m < N; ++m){
            itmp = (m <= n ? m : n);
            dtmp = 0;
            Li = L + m;
            Ui = U + n*ldu;
            for(i = 0; i <= itmp; ++i){
                if( i != m) dtmp += alpha * (*Li) * (*Ui);
                else dtmp += alpha * (*Ui);
                Li += ldl;
                ++Ui;
            }
            (*Ai) += dtmp;
            ++Ai;
        }
        Ai += (lda - N);
    }
    return 0;
}


int print_utpm(int P, int D, int ndim, int *A_shape, int *A_strides, double *A)
    /* print a string representation of the utpm instance A to the standard output
    
    only for 2D arrays implemented at the moment
    
    Parameters
    ----------
    P: int
        number of rays
    D: int
        D - 1 is the degree of the polynomial
    ndim: int
        number of dimensions
    A_shape: int array of size ndim
       shape of the ndarray A
    A_strides: int array of size ndim
       strides of the ndarray A
    
    A: double array of size prod(A_shape) + (D-1)*P * prod(A_shape)
    
    */

    {
    int i,size;
        
    if (ndim != 2){
        printf("only ndim=2 supported at the moment \n");
        return -1;
    }
    
    size = 1;
    for(i = 0; i < ndim; ++i){ size*=A_shape[i];}
        
    for(i = 0; i < (1+P*(D-1))*size; ++i ){
         printf("%f, ", A[i]);
    }
    printf("\n");
    
}



