#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

#define MIN(a, b) (a < b ? a : b)

__global__
void MatrixMulKernel(const int* M,const int* N, int* P, int Width) {
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((Row < Width) && (Col < Width))
	{
		int Pvalue = 0; 
		for (int k = 0; k < Width; ++k)
		{
			Pvalue += M[Row * Width + k] * N[k * Width + Col];
		}
		P[Row * Width + Col] = Pvalue;
	}

}

int main(){
	int *a, *g, *ag;
	int N = 1000;
	int thread = 256;
	dim3 threadsPerBlock(thread,thread);
	dim3 blocksPerGrid(MIN(32, (N + thread - 1) / thread), MIN(32, (N + thread - 1) / thread));

	cudaMallocManaged(&a,N * N * sizeof(int));
	cudaMallocManaged(&g, N * N * sizeof(int));
	cudaMallocManaged(&ag, N * N * sizeof(int));

	for(int i = 0; i < N; i++){
		for(int j = 0; j<N; j++){
			a[i*N+j] = 2; //było a[i*j+i] zamiast a[i*size+j]	aaaaaaaaaaaaaaaa kurde głupi błąd 
			g[i*N+j] = 3;
		}
	 }
	 MatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>( a, g, ag, N);
     cudaDeviceSynchronize();


//-----------------------------------------------------Writing out-------------------------------------------------------
	 for(int i = 0; i < N; i++){
		std:: cout<<"| ";
	 		for(int j = 0; j < N; j++){
	 			std::cout<<a[i * size + j]<<" ";
	 		}
			std:: cout<<"|"<<std::endl;
	 	}
	 std::cout<<"--------------------------------"<<std::endl;
	 for(int i = 0; i < N; i++){
		 std::cout<<"| ";
	 		for(int j = 0; j < N; j++){
	 			std::cout<<g[i * size + j]<<" ";
	 		}
			 std::cout<<"|"<<std::endl;
	 	}
	 std::cout<<"--------------------------------"<<std::endl;
	 for(int i = 0; i < N; i++){
		 std::cout<<"| ";
	 		for(int j = 0; j < N; j++){
	 			std::cout<<ag[i * size + j]<<" ";
	 		}
			 std::cout<<"|"<<std::endl;
	 	}
	    cudaFree(a);
	    cudaFree(g);
	    cudaFree(ag);
}	
