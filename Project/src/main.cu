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
#include <cuda_runtime.h>
#define BLOCK_SIZE 32

class CudaObject {

	int threadsPerBlock = 256;
	int blocksPerGrid = 32;
public:
	int size_x;
	int size_y;
	int *data;
	int stride;
	int bytes;
	int SM = 1;
	
	__host__ __device__
	CudaObject(int x, int y, int stride ): size_x(x), size_y(y),stride(stride){}
	
	__host__ __device__	
	CudaObject(const CudaObject &a): size_x(a.size_x), size_y(a.size_y),data(a.data),stride(a.stride){}
	
	__device__ 
	int getElement(int row, int col){
		return data[row * stride + col];
	}
	
	__host__ __device__
	void operator =(const CudaObject &a){size_x = a.size_x; size_y = a.size_y; data = a.data; stride = a.stride;}

	__device__
	 void setElement(int row, int col, int val){
		data[row * stride + col] = val;
	}
	
	__device__
	CudaObject cutMatrix(int row, int col){
		CudaObject tmp(BLOCK_SIZE, BLOCK_SIZE, stride);
		
		tmp.data = &data[stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
		return tmp;
	}
	
	__host__  
	 void writeOut(){
		for(int i = 0; i < size_x; i++){
			std::cout<<"| ";
		 	for(int j = 0; j < size_y; j++){
		 		std::cout<<data[i * size_y + j]<<" ";
		 	}
			std::cout<<"|"<<std::endl;
		 }
		std::cout<<"\n";
	}

	void setData(int *data, int x, int y){
		this->size_x = x;
		this->size_y = y;
		this->bytes = x * y * sizeof(int);
		cudaMallocManaged(&this->data, this->bytes);
		memcpy(this->data, data, this->bytes);
	}

	void setSize(int x, int y){
		this->size_x = x;
		this->size_y = y;
		this->bytes = x * y * sizeof(int);
		cudaMallocManaged(&this->data, this->bytes);
	}


	void addCpu(CudaObject &fData, CudaObject &sData){
		if(fData.size_x == sData.size_x && fData.size_y == sData.size_y){
			cudaMemPrefetchAsync(this->data, this->bytes, cudaCpuDeviceId);
			cudaMemPrefetchAsync(fData.data, fData.bytes, cudaCpuDeviceId);
			cudaMemPrefetchAsync(sData.data, sData.bytes, cudaCpuDeviceId);

			this->setSize(sData.size_x, sData.size_y);

			for(int i = 0; i < sData.size_x; i++){
				for(int j = 0; j < sData.size_y; j++){
					this->data[i * this->size_x + j] = fData.data[i * this->size_x + j] + sData.data[i * this->size_x + j];
				}
			}
		}
	}

	void subCpu(CudaObject &fData, CudaObject &sData){
		if(fData.size_x == sData.size_x && fData.size_y == sData.size_y){
			cudaMemPrefetchAsync(this->data, this->bytes, cudaCpuDeviceId);
			cudaMemPrefetchAsync(fData.data, fData.bytes, cudaCpuDeviceId);
			cudaMemPrefetchAsync(sData.data, sData.bytes, cudaCpuDeviceId);

			this->setSize(sData.size_x, sData.size_y);

			for(int i = 0; i < sData.size_x; i++){
				for(int j = 0; j < sData.size_y; j++){
					this->data[i * this->size_x + j] = fData.data[i * this->size_x + j] - sData.data[i * this->size_x + j];
				}
			}
		}
	}

	void mulCpu(CudaObject &fData, CudaObject &sData){
		int y_s = sData.size_y;
		int y_f = fData.size_y;
  		for(int i = 0; i < size_x; i++ ){
    		for(int j = 0; j < size_y; j++ ){
      			int s = 0;
      			for(int k = 0; k < y_f; k++ ) 
					s += fData.data[i * y_f + k] * sData.data[k * y_s + j];
				this->data[i * y_s + j] = s;
    		}
		}
	}

};

__global__ void add(int *fData, int *sData, int *oData, int x, int y){

		  int index = threadIdx.x + blockIdx.x * blockDim.x;
		  int stride = blockDim.x * gridDim.x;

		  for(int i = index; i < x*y; i += stride)
		  {
			oData[i] = fData[i] + sData[i];
		  }
}

__global__ void sub(int *fData, int *sData, int *oData, int x, int y){

	  int index = threadIdx.x + blockIdx.x * blockDim.x;
	  int stride = blockDim.x * gridDim.x;

	  for(int i = index; i < x*y; i += stride)
	  {
	    oData[i] = fData[i] - sData[i];
	  }
}	

__global__
void MatrixMulKernel(CudaObject a,CudaObject b, CudaObject c) {
	int cutRow = blockIdx.y ;
	int cutCol = blockIdx.x;

	int fRow = blockIdx.y * blockDim.y + threadIdx.y;
	int fCol = blockIdx.x * blockDim.x + threadIdx.x;	
	int row = threadIdx.y;
	int col = threadIdx.x;
	
	int temp = 0;
	
	CudaObject cutMatC = c.cutMatrix(cutRow, cutCol);
	
	for( int v = 0; v < ((a.size_y + BLOCK_SIZE - 1)/BLOCK_SIZE); ++v){
		CudaObject cutMatA = a.cutMatrix(cutRow, v);	//cut input matrix vector which can fit inside block
		CudaObject cutMatB = b.cutMatrix(v, cutCol);	
	
		__shared__ int A[BLOCK_SIZE][BLOCK_SIZE];	//Matrix wchich can share memory between threads
		__shared__ int B[BLOCK_SIZE][BLOCK_SIZE];
		if((row  < a.size_x) && ((col + v * BLOCK_SIZE) < a.size_y)){ 		
			A[row][col] = cutMatA.getElement(row, col);
		}
		else{
			A[row][col] = 0;
		}

		if((col < b.size_y) && ((row + v * BLOCK_SIZE) < b.size_x)){
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

	if(fRow < c.size_x && fCol < c.size_y)
		c.setElement(fRow, fCol, temp);
}

void OperationsInfo()
{
	std::cout<<"Choose an operation:"<<std::endl;
	std::cout<<"1. Matrix addition on CPU"<<std::endl;
	std::cout<<"2. Matrix addition on GPU"<<std::endl;
	std::cout<<"3. Matrix substraction on CPU"<<std::endl;
	std::cout<<"4. Matrix substraction on GPU"<<std::endl;
	std::cout<<"5. Matrix multiplication on CPU"<<std::endl;
	std::cout<<"6. Matrix multiplication on GPU"<<std::endl;
}

void Init(CudaObject &oData, int val)
{
	int x = oData.size_x;
	int y = oData.size_y;

	for(int i = 0; i < y; i++){
		for(int j = 0; j<x; j++){
			oData.data[i*x+j] = val;  
		}
	}
}

int main(){

	int operation;
	int N_1, N_2, M_1, M_2;
	int val_1, val_2;

	std::cout<<"Enter the values of size_x, size_y of the first matrix and value to filled matrix:"<<std::endl;
	std::cin>>N_1;
	std::cin>>M_1;
	std::cin>>val_1;

	std::cout<<"Enter the values of size_x, size_y of the second matrix and value to filled matrix::"<<std::endl;
	std::cin>>N_2;
	std::cin>>M_2;
	std::cin>>val_2;

	CudaObject fData(N_1, M_1, M_1), sData(N_2, M_2, M_2), oData(N_1, M_2, M_2);

	cudaMallocManaged(&fData.data,N_1 * M_1 * sizeof(int));
	cudaMallocManaged(&sData.data, N_2 * M_2 * sizeof(int));
	cudaMallocManaged(&oData.data, N_1 * M_2 * sizeof(int));

	Init(fData,val_1);
	Init(sData,val_2);

	fData.writeOut();	
	sData.writeOut();

	OperationsInfo();
	std::cin>>operation;

	dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 blocksPerGrid((fData.size_y + threadsPerBlock.x - 1)/threadsPerBlock.x, (sData.size_x + threadsPerBlock.y - 1)/threadsPerBlock.y);

	switch(operation)
	{
		case 1:
			if(fData.size_x != sData.size_x || fData.size_y != sData.size_y){
				std::cout<<"Matrices sizes have to be equal!"<<std::endl;
			}
			else{
				oData.addCpu(fData,sData);
				oData.writeOut();
			}
			break;
		case 2:
			if(fData.size_x != sData.size_x || fData.size_y != sData.size_y){
				std::cout<<"Matrices sizes have to be equal!"<<std::endl;
			}
			else{
				add<<<blocksPerGrid,threadsPerBlock>>>(fData.data, sData.data, oData.data, oData.size_x, oData.size_y);
				cudaDeviceSynchronize();
				oData.writeOut();
			}
			break;
		case 3:
			if(fData.size_x != sData.size_x || fData.size_y != sData.size_y){
				std::cout<<"Matrices sizes have to be equal!"<<std::endl;
			}
			else{
				oData.subCpu(fData,sData);
				oData.writeOut();
			}
			break;
		case 4:
			if(fData.size_x != sData.size_x || fData.size_y != sData.size_y){
				std::cout<<"Matrices sizes have to be equal!"<<std::endl;
			}
			else{
				sub<<<blocksPerGrid,threadsPerBlock>>>(fData.data, sData.data, oData.data, oData.size_x, oData.size_y);
				cudaDeviceSynchronize();
				oData.writeOut();
			}
			break;
		case 5:
			if(fData.size_y != sData.size_x){
				std::cout<<"Size_x of the first matrix and size_y of the second matrix have to be equal!"<<std::endl;
			}
			else{
				oData.mulCpu(fData,sData);
				oData.writeOut();
			}
			break;
		case 6:
			if(fData.size_y != sData.size_x){
				std::cout<<"Size_x of the first matrix and size_y of the second matrix have to be equal!"<<std::endl;
			}
			else{
				MatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(fData, sData, oData);	
				cudaDeviceSynchronize();
				oData.writeOut();
			}			
			break;

		default:
			std::cout<<"Wrong number entered!"<<std::endl;
			break;
	}
	
	cudaError_t err = cudaSuccess;	
		
	if (err != cudaSuccess){	
        	fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}
			
	cudaFree(fData.data);
	cudaFree(sData.data);
	cudaFree(oData.data);
}	
