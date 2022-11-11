#include "../gaussj_nr.h"

int main(){

    Matriz<double> a = Matriz<double>(2,2);
    Matriz<double> b = Matriz<double>(2,1);

    a[0][0] = 1;
    a[0][1] = 2;
    a[1][0] = 3;
    a[1][1] = -5;

    b[0][0] = 5;
    b[1][0] = 4;

    std::cout << "Matriz A:\n";
    a.print();

    std::cout << "Matriz B:\n";
    b.print();

    std::cout << "Aplicando algoritmo gaussj.h\n";
    gaussj(a,b);
    
    std::cout << "Matriz A:\n";
    a.print();

    std::cout << "Matriz B:\n";
    b.print();

    return 0;
}