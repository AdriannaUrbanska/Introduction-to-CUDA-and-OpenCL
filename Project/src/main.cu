#include <iostream>
#include <stdio.h>
#include <math.h>
#include "CudaObject.h"
#include <stdlib.h>

int main(void)
{
	int ** a;
	a = (int **)malloc(100 * sizeof(int *));
	for(int i = 0; i < 100; i++)
		a[i] = (int *)malloc(100 * sizeof(int)); 
	for(int i = 0; i < 100; i++)
		for(int j = 0; j < 100; j++)
			a[i][j] = i + j;
			
	CudaObject b = new CudaObject();
	for(int i = 0; i < 100; i++)
		free(a[i]);
	free(a);
 
	return 0;
}
