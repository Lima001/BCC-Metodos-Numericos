/*
    Eliminação Gaussiana com pivoteamento parcial
*/

#include "matriz.h"

template <class T> void gaussj(Matriz<T> &m){

    unsigned int i, im, j, k, l, nr=m.nLinha(), nc=m.nColuna();
    T aux, div;

    // Linha que está sendo considerada
    i = 0;

    // Percorrer colunas
    for (j=0; j<nc; j++){

        // Finalizar se todas as linhas já foram consideradas
        if (i == nr)
            break;

        // Indice da linha com o maior valor na coluna j - assume-se que seja linha i
        im = i;

        // Procurar a linha l com o maior valor na coluna j
        for (l=i+1; l<nr; l++){
            if (m[l][j] > m[im][j])
                im = l;
        }

        // Realizar a troca de linhas para que a linha com maior valor em j seja pivoteada
        if (im != i){
            for (l=0; l<nc; l++){
                aux = m[i][l];
                m[i][l] = m[im][l];
                m[im][l] = aux;
            }
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