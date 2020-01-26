
#ifndef CUDAOBJECT_H_
#define CUDAOBJECT_H_

#include <cuda_runtime.h>
#include <iostream>
//---------------------------------- Kernel Cuda Functions ---------------------------------

__global__ void add(int *fData, int *sData, int *oData, int x, int y);
__global__ void sub(int *fData, int *sData, int *oData, int x, int y);

class CudaObject{
//--------------------------------------- Values -------------------------------------------
	int threadsPerBlock;
	int blocksPerGrid;
public:
	int *data;
	int size_x;
	int size_y;
	int bytes;
	int deviceId;
	int SM;
//--------------------------------- Single Threaded Methods ---------------------------------
public:
	CudaObject(int *, int, int);
	CudaObject();
	~CudaObject();
	void addCpu(CudaObject &, CudaObject &);
	void addGpu(CudaObject &, CudaObject &);
	void subCpu(CudaObject &, CudaObject &);
	void subGpu(CudaObject &, CudaObject &);
	void show();
	void setData(int *, int x, int y);
	void setSize(int x, int y);

};


#endif /* CUDAOBJECT_H_ */

