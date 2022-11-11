#include <stdio.h>
#include <math.h>

#define ROOT_OF 2
#define ERROR 0.00000000001


double verify(double x){
	return x*x - ROOT_OF;
}

int main(){

	double min = 0;
	double max = ROOT_OF;
	double x = (max+min)/2.0;
	double v = verify(x);

	while (abs(v) > ERROR){

		printf("Min: %f / Max: %f\n", min, max);
		printf("x: %f / v: %f\n", x, v);

		if (v < 0)
			min = x;
		else
			max = x;
	
		x = (max+min)/2.0;
		v = verify(x);
	}
	
	printf("%f\n", v);
	printf("Raiz Aproximada de %d = %f\n", ROOT_OF, x);
	printf("Raiz %f\n", sqrt(ROOT_OF));

	return 0;
}
