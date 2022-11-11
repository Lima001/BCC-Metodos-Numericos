// Inclua o algoritmo que deseja utilizar

//#include "../gaussj_sp.h"
//#include "../gaussj_pp.h"
//#include "../gaussj_ppe.h"
//#include "../gaussj_tp.h"
#include "../gaussj_tpe.h"

int main(){

    Matriz<double> m = Matriz<double>(3,4);

    m[0][0] = 1;
    m[0][1] = 3;
    m[0][2] = -2;
    m[0][3] = 5;

    m[1][0] = 3;
    m[1][1] = 5;
    m[1][2] = 6;
    m[1][3] = 7;

    m[2][0] = 2;
    m[2][1] = 4;
    m[2][2] = 3;
    m[2][3] = 8;

    gaussj(m);
    m.print();

    return 0;
}