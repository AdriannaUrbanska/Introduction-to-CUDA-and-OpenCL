# CUDA Memory Managed Utility

In our second report we are going to answer three very important questions about CUDA Managed Memory Utility:

1. How large data structures are handled?
2. How do we use Memory Managed Utility?
3. How to protect against too large data structure to be copied to GPU?

### Question 1: How large data structures are handled?

### Question 2: How do we use Memory Managed Utility?

At first, we have to allocate Unified Memory, which is accesible from CPU or GPU, using cudaMallocManaged function. 
```
  cudaMallocManaged(&x, N*sizeof(float));
```


Our next step is to initialize x and y vectors on the host.
```
  for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
  }
```


Then, we launch CUDA kernel add function to add elements of two vectors and wait for GPU to finish before accessing on host.
```
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x);
  
  cudaDeviceSynchronize();
```


At the end, we have to delete memory using cudaFree function.
```
  cudaFree(x);
```

### Question 3: How to protect against too large data structure to be copied to GPU?

```
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  if( (unsigned long long) (N*sizeof(float)) >= (unsigned long long)deviceProp.totalGlobalMem) {
      fprintf(stderr, "Memory overload!\n");
      exit(EXIT_FAILURE);
  }
     
     
  if( threadsPerBlock >= deviceProp.maxThreadsPerBlock){
      fprintf(stderr, "Threads overload!\n");
      exit(EXIT_FAILURE);
   }

  if( blocksPerGrid >= deviceProp.maxGridSize[0]){
      fprintf(stderr, "Grid overload!\n");
    	exit(EXIT_FAILURE);
   }
```


## Authors

Adrianna Urbańska

Gabriel Chęć
