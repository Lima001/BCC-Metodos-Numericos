#include "../cholesky.h"

int main(){

    Matriz<double> m = Matriz<double>(4,4);
    Matriz<double> x = Matriz<double>(4,1);
    Matriz<double> b = Matriz<double>(4,1);

    m[0][0] = 4;
    m[0][1] = -2;
    m[0][2] = 4;
    m[0][3] = 2;

    m[1][0] = -2;
    m[1][1] = 10;
    m[1][2] = -2;
    m[1][3] = -7;

    m[2][0] = 4;
    m[2][1] = -2;
    m[2][2] = 8;
    m[2][3] = 4;

    m[3][0] = 2;
    m[3][1] = -7;
    m[3][2] = 4;
    m[3][3] = 7;

    b[0][0] = 8;
    b[1][0] = 2;
    b[2][0] = 16;
    b[3][0] = 6;

    /*
    Matriz<double> r = cholesky(m);

    std::cout << "Exibindo matriz r\n" << std::endl;
    r.print();
    */

    solve_cholesky(m,b,x);
    x.print();

    return 0;
}