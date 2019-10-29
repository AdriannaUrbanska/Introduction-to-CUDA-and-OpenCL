# CUDA Memory Managed Utility

In our second report we will answer three very important questions about CUDA Managed Memory Utility:

1. How large data structures are handled?
2. How do we use Memory Managed Utility?
3. How to protect against too large data structure to be copied to GPU?

### Question 1: How large data structures are handled?

### Question 2: How do we use Memory Managed Utility?

At first, we have to allocate Unified Memory, which is accesible from CPU or GPU, using cudaMallocManaged function. 
```
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
```


Our next step is to initialize x and y vectors on the host.
```
  for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
  }
```


Then, we launch CUDA kernel add function to add elements of two vectors and wait for GPU to finish before accessing on host.
```
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x, y);
  
  cudaDeviceSynchronize();
```


At the end, we have to delete memory using cudaFree function.
```
  cudaFree(x);
  cudaFree(y);
```

### Question 3: How to protect against too large data structure to be copied to GPU?




## Authors

Adrianna Urbańska

Gabriel Chęć
