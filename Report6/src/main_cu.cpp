#include <iostream>
#define N 16777216
#include <stdlib.h>
#include <ctime>

int main(){
	int* tab;
	tab = (int*)malloc(N*sizeof(int));
	for(int i = 0; i < N;i++)
		tab[i] = 1;
	int eq=0;
	for(int i = 0; i<N;i++)
		eq+=tab[i];
std::cout<<"Wynik: "<<eq<<std::endl;
	free(tab);
}


