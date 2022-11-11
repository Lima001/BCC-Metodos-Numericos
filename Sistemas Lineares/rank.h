#include "rref.h"

template <class T> unsigned int rank(Matriz<T> &m){

    Matriz<T> copy = m;
    unsigned int r=0;
    
    /* 
        rref
    */
    
    unsigned int i, j, k, l, a, nr=copy.nLinha(), nc=copy.nColuna();
    T tmp, div, mul;

    // Linha a ser pivoteada/reduzida
    i = 0;

    // Percorrer colunas
    for (j=0; j<nc; j++){

        // Linha que contém a coluna j != 0 (Assumimos incialmente que seja a linha i)
        k = i;

        // Encontrar a linha cuja coluna j != 0
        while (k < nr && copy[k][j] == 0)
            k++;

        // Se nenhuma linha possui a coluna j != 0, prossiga para a próxima coluna
        if (k == nr)
            continue;

        // Se a linha encontrada for diferente da assumida inicialmente, fazer a troca de linhas
        if (k != i){
            for (l=0; l<nc; l++){
                tmp = copy[i][l];
                copy[i][l] = copy[k][l];
                copy[k][l] = tmp;
            }
        }

        // Escalonar a linha em questão para produzir um pivô em j
        div = 1/copy[i][j];
        for (l=0; l<nc; l++)
            copy[i][l] *= div;

        // Incrementar o rank da matriz
        // Observe que na execução do laço anterior, cria-se um pivô...
        // Por isso pode-se incrementar o rank em uma unidade
        r++;

        // Realizar a redução das demais linhas com base na linha anterior
        for (a=0; a<nr; a++){
            if (a == i)
                continue;
            
            mul = copy[a][j];
            for (l=0; l<nc; l++){
                copy[a][l] = copy[a][l] - copy[i][l]*mul;
            }
        }

        // Avançar para a próxima linha
        i++;
    }
    return r;
}