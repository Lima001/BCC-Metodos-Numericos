#include <assert.h>
#include <cmath>

#include "matriz.h"

template <class T> Matriz<T> cholesky(Matriz<T> &m){

    int i, j, k, r=m.nLinha(), c=m.nColuna();

    assert(r == c);

    Matriz<T> mr = Matriz<T>(r,c);
    T sum;

    for (i=0; i<r; i++){
        
        for (j=i; j<c; j++){
            
            for (sum=m[i][j], k=i-1; k>=0; k--)
                sum -= mr[i][k] * mr[j][k];

            if (i == j){
                if (sum <= T(0)){
                    mr.~Matriz();
                    std::cerr << "A matriz informada não é definida positiva" << std::endl;
                    exit(1);
                }
                mr[i][i] = std::sqrt(sum);
            }
            else
                mr[j][i] = sum/mr[i][i];
        }
    }
    return mr;
}


template <class T> void solve_cholesky(Matriz<T> &m, Matriz<T> &b, Matriz<T> &x){

    assert(m.nColuna() == b.nLinha() && m.nColuna() == x.nLinha());

    Matriz<T> mr = cholesky(m);
    int i, k, n=m.nColuna();
    T sum;

    for (i=0; i<n; i++){
        for (sum=b[i][0], k=i-1; k>=0; k--)
            sum -= mr[i][k] * x[k][0];
        
        x[i][0] = sum/mr[i][i];
    }

    for (i=n-1; i>=0; i--){
        for (sum=x[i][0], k=i+1; k<n; k++)
            sum -= mr[k][i] * x[k][0];
        
        x[i][0] = sum/mr[i][i];
    }
}