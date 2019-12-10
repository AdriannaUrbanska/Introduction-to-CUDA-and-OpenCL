#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#define N 511
#define BLOCK_SIZE 16

__global__ void  reduceSum(int *ada, int *gabrys){
	__shared__ int partialSum[2 * BLOCK_SIZE];
	unsigned int t = threadIdx.x;
	unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;

	if(start + t < N){
		partialSum[t] = ada[start + t];
	}
	else{
		partialSum[t] = 0;
	}

	if (start + BLOCK_SIZE + t < N){
		partialSum[BLOCK_SIZE + t] = ada[start + BLOCK_SIZE + t];
	}
	else{
		partialSum[BLOCK_SIZE + t] = 0;
	}   

	for(unsigned int stride = 1; stride <= BLOCK_SIZE ; stride *= 2){
		__syncthreads();
		if (t % stride == 0 ){
			partialSum[2*t] += partialSum[2*t + stride];
		}	
	} 

	if(t == 0){
		gabrys[blockIdx.x] = partialSum[0];
	}

	__syncthreads();
}

int main(void){
	int * ada, * gabrys;
	cudaMallocManaged(&ada, N * sizeof(int));
	cudaMallocManaged(&gabrys, N * sizeof(int));

	for(int i = 0; i < N; i++){
		ada[i] = 1;
	}
	dim3 threadsPerBlock(BLOCK_SIZE);
	dim3 blocksPerGrid((N + BLOCK_SIZE - 1)/BLOCK_SIZE);
		
	reduceSum<<<blocksPerGrid, threadsPerBlock>>>(ada,gabrys);
	cudaDeviceSynchronize();
	
	int count = 1;

	for(int i = 0; i<count; i++){
		reduceSum<<<blocksPerGrid, BLOCK_SIZE>>>(gabrys,gabrys);
		cudaDeviceSynchronize();
	}
	
	for(int i = 0; i<10; i++)
	std::cout<<gabrys[i]<<std::endl;
}
