#include "cassert"
#include "matriz.h"

template <class T> void rref(Matriz<T> &m){

    unsigned int i, j, k, l, a, nr=m.nLinha(), nc=m.nColuna();
    T tmp, div, mul;

    // Linha a ser pivoteada/reduzida
    i = 0;

    // Percorrer colunas
    for (j=0; j<nc; j++){

        // Linha que contém a coluna j != 0 (Assumimos incialmente que seja a linha i)
        k = i;

        // Encontrar a linha cuja coluna j != 0
        while (k < nr && m[k][j] == 0)
            k++;

        // Se nenhuma linha possui a coluna j != 0, prossiga para a próxima coluna
        if (k == nr)
            continue;

        // Se a linha encontrada for diferente da assumida inicialmente, fazer a troca de linhas
        if (k != i){
            for (l=0; l<nc; l++){
                tmp = m[i][l];
                m[i][l] = m[k][l];
                m[k][l] = tmp;
            }
        }

        // Escalonar a linha em questão para produzir um pivô em j
        div = 1/m[i][j];
        for (l=0; l<nc; l++)
            m[i][l] *= div;

        // Realizar a redução das demais linhas com base na linha anterior
        for (a=0; a<nr; a++){
            if (a == i)
                continue;
            
            mul = m[a][j];
            for (l=0; l<nc; l++){
                m[a][l] = m[a][l] - m[i][l]*mul;
            }
        }

    // Avançar para a próxima linha
        i++;
    }

    return;
}