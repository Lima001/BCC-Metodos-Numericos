/*
    Encontrar epsilon (e) da máquina

    O 'e' da máquina refere-se ao menor número que
    não é arredondado para zero.
*/

#include <iostream>
#include <limits>
#include <cmath>

int main(){

    double e = -1;

    while (-1+e < -1)
        e = e/2;

    std::cout << e << std::endl;
    std::cout << (fabs(e) <= std::numeric_limits<double>::epsilon()) << std::endl;

    return 0;
}