/*
    Eliminação Gaussiana com pivoteamento completo
*/

#include "matriz.h"

template <class T> void gaussj(Matriz<T> &m){

    unsigned int i, im, j, jm, k, l, nr=m.nLinha(), nc=m.nColuna();
    T aux, div;

    // Informações para reverter o impacto da troca de colunas
    unsigned int pos_troca[nr*2];   // Array que guarda as colunas que foram alteradas (de dois em dois indices)
    unsigned int ipos_troca = 0;    // Indice para manusear o array anterior

    // Inserir valores padrões no array. 0 foi escolhido para representar que não houve troca de colunas.
    // Sendo assim, quando ocorrer uma troca do tipo coluna 1 pela coluna 2 (lembrando que os indices da matriz
    // começam em 0), o array em questão guarda a informações 2 e 3 em dois espaços da memória. Por fim,
    // ao reverter o impacto das trocas de colunas, deve-se observar esse fato e descontar 1 para obter o valor
    // real da coluna na matriz.
    for (i=0; i<nr*2; i++)
        pos_troca[i] = 0;

    // Linha que está sendo considerada
    i = 0;

    // Percorrer colunas
    for (j=0; j<nc; j++){
        m.print();
        std::cout << std::endl;
        // Finalizar se todas as linhas já foram consideradas
        if (i == nr)
            break;

        // Indice da linha com o maior valor na coluna j - assume-se que seja linha i
        im = i;
        jm = j;

        // Procurar a linha/coluna com o maior valor visando pivoteamento
        for (l=i; l<nr; l++){
            for (k=j; k<nr; k++){               // Percorrer colunas até ordem quadrada
                if (m[l][k] > m[im][jm]){
                    im = l;
                    jm = k;
                }
            }
        }

        // Trocar coluna
        if (jm != j){
            for (l=0; l<nr; l++){
                aux = m[l][j];
                m[l][j] = m[l][jm];
                m[l][jm] = aux;
            }
            // Armazena (conforme padrão explicado anteriormente) o indice das colunas trocadas
            pos_troca[ipos_troca] = j+1;
            pos_troca[++ipos_troca] = jm+1;
        }
        m.print();
        std::cout << std::endl;

        // Trocar linha
        if (im != i){
            for (l=0; l<nc; l++){
                aux = m[i][l];
                m[i][l] = m[im][l];
                m[im][l] = aux;
            }
        }

        m.print();
        std::cout << std::endl;
        std::cout << "--------------" << std::endl;

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

    // Reverte o efeito da troca de colunas
    ipos_troca = 0;
    while (ipos_troca < nr*2 && pos_troca[ipos_troca] != 0){
        for (l=0; l<nc; l++){
            aux = m[pos_troca[ipos_troca]-1][l];
            m[pos_troca[ipos_troca]-1][l] = m[pos_troca[ipos_troca+1]-1][l];
            m[pos_troca[ipos_troca+1]-1][l] = aux;
        }
        ipos_troca += 2;
    }

    return;
}