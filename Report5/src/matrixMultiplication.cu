#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#define MIN(a, b) (a<b?a:b)
#define BLOCK_SIZE 32

struct Matrix {
	int height;
	int width;
	int *el;
	int stride;
	
	__host__ __device__
	Matrix(int height, int width, int stride ): height(height), width(width),stride(stride){}
	
	__host__ __device__	
	Matrix(const Matrix &a): height(a.height), width(a.width),el(a.el),stride(a.stride){}
	
	__device__ 
	float getElement(int row, int col){
		return el[row * stride + col];
	}
	
	__host__ __device__
	void operator =(const Matrix &a){height = a.height; width = a.width; el = a.el; stride = a.stride;}

	__device__
	 void setElement(int row, int col, int val){
		el[row * stride + col] = val;
	}
	
	__device__
	 Matrix cutMatrix(int row, int col){
		Matrix tmp(BLOCK_SIZE, BLOCK_SIZE, stride);
		
		tmp.el = &el[stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
		return tmp;
	}
	
	__host__  
	 void writeOut(){
		for(int i = 0; i < height; i++){
			std::cout<<"| ";
		 	for(int j = 0; j < width; j++){
		 		std::cout<<el[i * width + j]<<" ";
		 	}
			std::cout<<"|"<<std::endl;
		 }
		std::cout<<"\n";
	}
};


__global__
void MatrixMulKernel(Matrix a,Matrix b, Matrix c) {
	int cutRow = blockIdx.y ;
	int cutCol = blockIdx.x;

	int fRow = blockIdx.y * blockDim.y + threadIdx.y;
	int fCol = blockIdx.x * blockDim.x + threadIdx.x;	
	int row = threadIdx.y;
	int col = threadIdx.x;
	
	int temp = 0;
	
	Matrix cutMatC = c.cutMatrix(cutRow, cutCol);
	
	for( int v = 0; v < ((a.width + BLOCK_SIZE - 1)/BLOCK_SIZE); ++v){
		Matrix cutMatA = a.cutMatrix(cutRow, v);	//cut input matrix vector which can fit inside block
		Matrix cutMatB = b.cutMatrix(v, cutCol);	
	
		__shared__ int A[BLOCK_SIZE][BLOCK_SIZE];	//Matrix wchich can share memory between threads
		__shared__ int B[BLOCK_SIZE][BLOCK_SIZE];
		if((row  < a.height) && ((col + v * BLOCK_SIZE) < a.width)){ 		
			A[row][col] = cutMatA.getElement(row, col);
		}
		else{
			A[row][col] = 0;
		}

		if((col < b.width) && ((row + v * BLOCK_SIZE) < b.height)){
			B[row][col] = cutMatB.getElement(row, col);
		}
		else{
			B[row][col] = 0;
		}

		__syncthreads();				//make sure that every metrix is filled
	
		for (int i = 0; i < BLOCK_SIZE; ++i){
			temp += A[row][i] * B[i][col];
		}		
		__syncthreads();
	
	}

	if(fRow < c.height && fCol < c.width)
		c.setElement(fRow, fCol, temp);
}

int main(){
	int N = 12;
	int M = 8;
	Matrix a(N, M, M), g(M, N, N), ag(N, N, N);
	
	cudaError_t err = cudaSuccess;	
	
	dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 blocksPerGrid((a.width + threadsPerBlock.x - 1)  / threadsPerBlock.x ,(g.height + threadsPerBlock.y - 1) / threadsPerBlock.y  );
	
	cudaMallocManaged(&a.el,N * M * sizeof(int));
	cudaMallocManaged(&g.el, N * M * sizeof(int));
	cudaMallocManaged(&ag.el, N * N * sizeof(int));
	
	for(int i = 0; i < M; i++){
		for(int j = 0; j<N; j++){
			a.el[i*N+j] = 1;  
			g.el[i*N+j] = 2;
		}
	}

	MatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>( a, g, ag);
	
     	cudaDeviceSynchronize();
	if (err != cudaSuccess){	
        	fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}
	
	a.writeOut();	
	g.writeOut();
	ag.writeOut();

			
	cudaFree(a.el);
	cudaFree(g.el);
	cudaFree(ag.el);
}	
