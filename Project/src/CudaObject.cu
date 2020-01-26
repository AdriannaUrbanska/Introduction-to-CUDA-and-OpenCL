#include "CudaObject.h"
#include <iostream>
#include <stdio.h>


CudaObject::CudaObject( int * data, int x, int y){
	this->size_x = x;
	this->size_y = y;
	this->bytes = x * y * sizeof(int);
	cudaMallocManaged(&this->data, this->bytes);
	memcpy(this->data, data, this->bytes);
	cudaGetDevice(&this->deviceId);
	cudaDeviceGetAttribute(&this->SM, cudaDevAttrMultiProcessorCount,this->deviceId);
	cudaGetDevice(&this->deviceId);
	this->threadsPerBlock = 256;
	this->blocksPerGrid = 32 * this->SM;
}

CudaObject::CudaObject(){
	this->size_x = 0;
	this->size_y = 0;
	this->bytes = 0;
	this->data = 0;

	cudaGetDevice(&this->deviceId);
	cudaDeviceGetAttribute(&this->SM, cudaDevAttrMultiProcessorCount,this->deviceId);
	this->threadsPerBlock = 256;
		this->blocksPerGrid = 32 * this->SM;

}

CudaObject::~CudaObject() { cudaFree(this->data);}

void CudaObject::setData(int *data, int x, int y){
		this->size_x = x;
		this->size_y = y;
		this->bytes = x * y * sizeof(int);
		cudaMallocManaged(&this->data, this->bytes);
		memcpy(this->data, data, this->bytes);
}

void CudaObject::setSize(int x, int y){
		this->size_x = x;
		this->size_y = y;
		this->bytes = x * y * sizeof(int);
		cudaMallocManaged(&this->data, this->bytes);
}

void CudaObject::addGpu(CudaObject &fData, CudaObject &sData){
	if(fData.size_x == sData.size_x && fData.size_y == sData.size_y){
		this->setSize(sData.size_x, sData.size_y);

		cudaMemPrefetchAsync(this->data, this->bytes, this->deviceId);
		cudaMemPrefetchAsync(sData.data, sData.bytes, this->deviceId);
		cudaMemPrefetchAsync(fData.data, fData.bytes, this->deviceId);
		add<<<this->blocksPerGrid, this->threadsPerBlock>>>(fData.data, sData.data, this->data,sData.size_x, sData.size_y);

		cudaDeviceSynchronize();
	}
}


void CudaObject::addCpu( CudaObject &fData, CudaObject &sData){
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

void CudaObject::subGpu(CudaObject &fData, CudaObject &sData){
	if(fData.size_x == sData.size_x && fData.size_y == sData.size_y){
		this->setSize(sData.size_x, sData.size_y);

		cudaMemPrefetchAsync(this->data, this->bytes, this->deviceId);
		cudaMemPrefetchAsync(sData.data, sData.bytes, this->deviceId);
		cudaMemPrefetchAsync(fData.data, fData.bytes, this->deviceId);
		sub<<<this->blocksPerGrid, this->threadsPerBlock>>>(fData.data, sData.data, this->data,sData.size_x, sData.size_y);

		cudaDeviceSynchronize();
	}
}


void CudaObject::subCpu( CudaObject &fData, CudaObject &sData){
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

void CudaObject::show(){
	cudaMemPrefetchAsync(this->data, this->bytes, cudaCpuDeviceId);

	for(int i = 0; i < this->size_x; i++){
		std::cout<<"|\t";
		for(int j = 0; j < this->size_y; j++){
			std::cout<<data[i * this->size_x + j]<< "\t";
		}
		std::cout<<" |"<<std::endl;
	}
}

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

