# Matrix Multiplication (NxN)

In this report we wrote a code with matrix multiplication algorithm where in every matrix its size of width was equal to the size of height (NxN). To make our program run faster we used shared memory buffer.

## Code

In our code ([matrixMultiplication](https://github.com/AdriannaUrbanska/Introduction-to-CUDA-and-OpenCL/blob/master/Report4/src/matrixMultiplication.cu)) we created a stuct Matrix which contains:

* information about matrix parameters
* 2 constructors (one of them is the copy constructor) and assignment operator - available from both the host and the device
* writeOut() function to print elements of the matrix - available from the host
* functions: setElement(int,int,int) - to set value of a specific element, getElement(int,int) - to get value of a specific element, cutMatrix(int,int) - to get a part of the matrix (new smaller matrix which have width and height equal to BLOCK_SIZE)- all available from the device

Then we have __global__ void MatrixMulKernel(Matrix,Matrix,Matrix) function.


## Authors

Adrianna Urbańska

Gabriel Chęć
