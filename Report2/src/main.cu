    #include <iostream>
    #include <stdio.h>
    #include <math.h>


    // CUDA kernel to add elements
    __global__    void add(int N, float *x)
    {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
   
      if (i<N) 
          x[i] = x[i] *2;
    }

    int main(void)
    {
      int N = 1<<20;
      float *x;

      // Allocate Unified Memory -- accessible from CPU or GPU
      cudaMallocManaged(&x, N*sizeof(float));

      // initialize x array on the host
      for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
      }

      // Launch kernel on 1M elements on the GPU
      int threadsPerBlock = 256;
      int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

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
      
      add<<<threadsPerBlock, blocksPerGrid >>>(N, x);

      // Wait for GPU to finish before accessing on host
      cudaDeviceSynchronize();

      // Free memory
      cudaFree(x);

      return 0;
    }
