#include "../matriz.h"

using namespace std;

int main(){

    // Criação de uma Matriz 3x3
    Matriz<double> m1 = Matriz<double>(3,3);

    // Definição de linhas para a matriz
    vector<double> l1 = {1,1,1};
    vector<double> l2 = {2,2,2};
    vector<double> l3 = {3,3,3};

    m1.definir_linha(0,l1);
    m1.definir_linha(1,l2);
    m1.definir_linha(2,l3);

    // Criação de uma nova Matriz por cópia
    Matriz<double> m2 = m1;

    // Exibição da Matriz
    m1.print();
    cout << endl;
    
    // Testes com as operações de Matrizes com escalar
    (m1 *= 2).print();
    cout << endl;
    (m1 += m2).print();
    cout << endl;
    (m1 + m2*2).print();
    cout << endl;
    (2*m1 + m2).print();
    cout << endl;
    (m1 -= m2).print();
    cout << endl;
    (m1 - m2).print();

    cout << endl;

    // Acesso, Recuperação e Alteração de elementos usando []
    cout << m1[0][0] << endl;
    m1[0][0] = 57;
    cout << m1[0][0] << endl;
    m1.print();

    cout << endl;

    // Criação de Matrizes para testes
    Matriz<double> A = Matriz<double>(3,2);
    Matriz<double> B = Matriz<double>(2,3);

    A.definir_linha(0, vector<double>{2,3});
    A.definir_linha(1, vector<double>{0,1});
    A.definir_linha(2, vector<double>{-1,4});

    B.definir_linha(0, vector<double>{1,2,3});
    B.definir_linha(1, vector<double>{-2,0,4});

    // Testes com as operações de Matrizes com escalar
    (A*B).print();
    cout << endl;
    (B*A).print();

    return 0;
}