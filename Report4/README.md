# Matrix Multiplication (NxN)

In this report we wrote a code with matrix multiplication algorithm where in every matrix its size of width was equal to the size of height (NxN). To make our program run faster we used shared memory buffer.

## Code

In our code ([matrixMultiplication](https://github.com/AdriannaUrbanska/Introduction-to-CUDA-and-OpenCL/blob/master/Report4/src/matrixMultiplication.cu)) we created a stuct Matrix which contains:

* information about matrix parameters
* 2 constructors (one of them is the copy constructor) and assignment operator - available from both the host and the device
* writeOut() function to print elements of the matrix - available from the host
* functions: setElement(int,int,int) - to set value of a specific element, getElement(int,int) - to get value of a specific element, cutMatrix(int,int) - to get a part of the matrix (new smaller matrix which have width and height equal to BLOCK_SIZE)- all available from the device

Then we have `__global__ void MatrixMulKernel(Matrix a, Matrix b,Matrix c)` function. At first, we created a `cutMatC` matrix as a part of resulting matrix `c`.

```
	Matrix cutMatC = c.cutMatrix(cutRow, cutCol);
```

Next we created a for loop where a main part of the multiplication takes part. In every execution of the loop we created a `cutMatA` and `cutMatB` matrices to keep a specific part of the origin matices `a` and `b`. Then we created `A`, `B` matrices which can share memory between threads:

```
	__shared__ int A[BLOCK_SIZE][BLOCK_SIZE];	
	__shared__ int B[BLOCK_SIZE][BLOCK_SIZE];
```

Then we checked if the threads indexes were still in the origin matrices area. If the conditions were fulfiled we filled matreces `A` and `B` with proper values from `cutMatA` and `cutMatB` matrices. After that operation we used `__syncthreads()` function to make sure that every matrix is filled. 

```
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

	__syncthreads(); 
```

One of the last things was to sum up the results from multiplying the proper elements from the row from matrix A and a column from matrix B and assigned the result to `temp` variable. Atfer that we used `__syncthreads()` function again.

```
	for (int i = 0; i < BLOCK_SIZE; ++i){
		temp += A[row][i] * B[i][col];
	}		
	__syncthreads();
```

At the end of the whole function, if the threads were still in the matrix `c` area, we assigned `temp` variable to a specific element of the resulting matrix `c`.

```
	c.setElement(fRow, fCol, temp);
```

In `main()` function we created `a`,`g` and `ag` matrices (the last one is the result of the multiplication) and allocated memory for them using `cudaMallaocManaged` function. Every matrix has the same size here.

```
	Matrix a(N, N, N), g(N, N, N), ag(N, N, N);
```

We launched kernel `MatrixMulKernel` function for (BLOCK_SIZE, BLOCK_SIZE) thread per block and `( (N + BLOCK_SIZE - 1)/BLOCK_SIZE, (N + BLOCK_SIZE - 1)/BLOCK_SIZE )` blocks per grid and synchronized threads using `cudaDeviceSynchronize()`.

```
	dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE ,(N + BLOCK_SIZE - 1) / BLOCK_SIZE );

	//...	
	MatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>( a, g, ag);
	cudaDeviceSynchronize();
```

## Results from nvprof tool

### Dependence of the execution time on BLOCK_SIZE for 64x64 matrix size

![alt text](https://github.com/AdriannaUrbanska/Introduction-to-CUDA-and-OpenCL/blob/master/Report4/img/block_size3.png)

As we can see on the plot the best results are when we executing our program with BLOCK_SIZE equal to 16 or 32.

### Dependence of the execution time on matrix size for BLOCK_SIZE = 32

![alt text](https://github.com/AdriannaUrbanska/Introduction-to-CUDA-and-OpenCL/blob/master/Report4/img/matrix_size2.png)


## Authors

Adrianna Urbańska

Gabriel Chęć
