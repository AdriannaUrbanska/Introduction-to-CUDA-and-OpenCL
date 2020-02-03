# A simple calculator for matrices - CUDA Project

In order to summarize the knowledge acquired during the CUDA course we prepared a simple application for matrices.
We created four operations, which can be performed both on CPU and GPU
1. Addition of the matrices in the same size
2. Substraction of the two matrices in the same size
3. Multiplication of the two matrices where the size of column of the first one is equal to the size of rows of the second one
4. Tranposition of matrix

## Code

At the beginning, we created CudaObject class, which contains properties of a single matrix:

```
class CudaObject {

public:
	int size_x;   // size of the rows
	int size_y;   // size of the columns
	int *data;    // pointer which contains elements of the matrix
	int stride;   // stride number
	int bytes;    // number of bytes to allocate for data pointer
	
	__host__ __device__
	CudaObject(int x, int y, int stride);         //constructor with three int parameters
	
	__host__ __device__	
	CudaObject(const CudaObject &a);              //copy constructor
	
	__device__ 
	int getElement(int row, int col);             //method which returns one element of the matrix
  
	__device__
	CudaObject cutMatrix(int row, int col);       //method which divides and returns part of the matrix
	 
  __device__
	void setElement(int row, int col, int val);   //method which sets value of the element of the matrix
	void setSize(int x, int y);                   //method which sets size of size_x and size of the size_y
  
	void addCpu(CudaObject &fData, CudaObject &sData);  //Addition of the matrices on the CPU
	void subCpu(CudaObject &fData, CudaObject &sData);  //Substraction of th ematrices on the CPU
	void mulCpu(CudaObject &fData, CudaObject &sData);  //Multiplication of th ematrices on the CPU
	void tranCpu(CudaObject &iData);                    //Transposition of th ematrices on the CPU
  
  __host__  
	void writeOut();                              //method which prints the matrix
};
```

Then, we create kernels for operations on the GPU:
```
__global__ void add(int *fData, int *sData, int *oData, int x, int y); 	//Addition of the matrices on the GPU
__global__ void sub(int *fData, int *sData, int *oData, int x, int y); 	//Substraction of th ematrices on the GPU
__global__ void mul(CudaObject a,CudaObject b, CudaObject c); 		//Multiplication of th ematrices on the GPU
__global__ void tran(CudaObject iData, CudaObject oData); 		//Transposition of th ematrices on the GPU
```

In 'main()' function we provided a possibilities for user to enter sizes of the two matrices and values for initialization and to choose which operation to performance:

```
Enter the values of size_x, size_y of the first matrix and value to filled matrix:
>
>
>

Enter the values of size_x, size_y of the second matrix and value to filled matrix:
>
>
>

Choose an operation:
1. Matrix addition on CPU
2. Matrix addition on GPU
3. Matrix substraction on CPU
4. Matrix substraction on GPU
5. Matrix multiplication on CPU
6. Matrix multiplication on GPU
7. Matrix transposition on CPU
8. Matrix transposition on GPU
>
```

## Authors

Adrianna Urbańska

Gabriel Chęć
