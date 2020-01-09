#include "CudaObject.h"
#include <iostream>

CudaObject::CudaObject( T * data, int x, int y): data( data ) {
	this.size.x = x;
	this.size.y = y;
}

void CudaObject::add( T * secondData){

}

CudaObject::CudaObject(){
	cudaFree(this.data);
}

void CudaObject::show(){
	for(int i = 0; i < this.size.x; i++){
		std::cout<<"|\t";
		for(int j = 0; j < this.size.y; j++){
			std::cout<<data[i][j]<< "\t";
		}
		std::cout<<"\t |"<<std::endl;
	}
}
