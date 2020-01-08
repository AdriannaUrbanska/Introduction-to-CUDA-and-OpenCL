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
	Manager( T ** );
	void add( T ** );
	void subtrack( T ** );
	void miltiply( T ** );
	void inverse( );
	void eigenValues( );

	
//-------------------------------------- Cuda Methods ---------------------------------------
	__global__ void add( T ** );
	__global__ void subtrack( T ** );
	__global__ void miltiply( T ** );
	__global__ void inverse( );
	__global__ void eigenValues( );


};
