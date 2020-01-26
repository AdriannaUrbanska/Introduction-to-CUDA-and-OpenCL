/*
 ============================================================================
 Name        : CudaProject.cu
 Author      : Adrianna Urbańska, Gabriel Chęć
 Version     :
 Copyright   : 
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <stdio.h>
#include <math.h>
#include "CudaObject.h"
#include <stdlib.h>

int main(void)
{
	int * a, *b;
	int size = 2<<14;
	size_t s = size * size * sizeof(int);

	a = (int *)malloc(s);
	b = (int *)malloc(s);

	//cudaMallocManaged(&a, s);
	//cudaMallocManaged(&b, s);

	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			a[i * size + j] = 2;
			b[i * size + j] = 3;
		}
	}

	CudaObject a_mat(a, size, size);
	CudaObject b_mat(b, size, size);
	CudaObject c_mat;

	free(b);
	free(a);
	cudaFree(a);
	cudaFree(b);
	c_mat.addGpu(a_mat,b_mat);
	//c_mat.show();
	return 0;
}
