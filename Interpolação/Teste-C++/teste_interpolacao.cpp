#include <iostream>
#include "interpolacao.h"

void teste_linear(){
    double xl, x0, x1, y0, y1;
    x0 = 0;
    x1 = 1;
    y0 = -1.0/2.0;
    y1 = 5.0/2.0;

    xl = 1.0/2.0;

    std::cout << interp_lin(xl, x0, x1, y0, y1) << std::endl;
}

void teste_bilinear(){
    double xl, yl, x0, x1, y0, y1, fx0y0, fx0y1, fx1y0, fx1y1;
    
    x0 = 0;
    x1 = 5;
    y0 = 0;
    y1 = 5;

    fx0y0 = 3.3;
    fx0y1 = 2.9;
    fx1y0 = 3.1;
    fx1y1 = 0.6;

    xl = 0.7;
    yl = 3.9;

    std::cout << interp_bilin(xl, yl, x0, x1, y0, y1, fx0y0, fx0y1, fx1y0, fx1y1) << std::endl;
}

int main(){

    //teste_linear();
    teste_bilinear();

    return 0;
}