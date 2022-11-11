/*
    Eliminação Gaussiana sem pivoteamento
*/

#include "matriz.h"

template <class T> void gaussj(Matriz<T> &m){

    unsigned int i,j, k, l, nr=m.nLinha(), nc=m.nColuna();
    T aux, div;

    // Linha que está sendo considerada
    i = 0;

    // Percorrer colunas
    for (j=0; j<nc; j++){

        if (i == nr)
            break;

        // Se pivô = 0, finalizar algoritmo? Continuar próxima coluna?
        if (m[i][j] == 0){
            break;
            //i++;
            //continue;
        }

        // Realiza o escalonamento da linha i com base no valor da coluna j
        div = 1/m[i][j];
        for (k=0; k<nc; k++)
            m[i][k] *= div;

        // Realiza a redução das linhas com base na linha i
        for (l=0; l<nr; l++){
            if (l == i)
                continue;
            
            aux = m[l][j];
            for (k=0; k<nc; k++)
                m[l][k] = m[l][k] - aux*m[i][k];
        }

        i++;
    }

    return;
}