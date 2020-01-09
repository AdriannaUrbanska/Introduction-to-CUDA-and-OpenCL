#include <iostream>

struct size_struct{
	int x;
	int y;
}

template <class T>
class CudaObject{
//--------------------------------------- Values -------------------------------------------
	T **data; 
	dim3 threadsperBlock;
	dim3 blocksPerGrid;
	size_struct size;

//--------------------------------- Single Threaded Methods ---------------------------------
public:
	CudaObject( T **,int, int );
	~CudaObject()
	void add( T ** );
	void subtrack( T ** );
	void miltiply( T ** );
	void inverse( );
	void eigenValues( );
	void show();
	
//-------------------------------------- Cuda Methods ---------------------------------------
	__global__ void c_add( T ** );
	__global__ void c_subtrack( T ** );
	__global__ void c_multiply( T ** );
	__global__ void c_inverse( );
	__global__ void c_eigenValues( );
};
