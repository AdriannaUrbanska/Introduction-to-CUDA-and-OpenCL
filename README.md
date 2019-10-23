# CUDA functions execution dependencies

  In this report we research dependencies of execution time on functions in CUDA library. 
  We measured execution time according to:
  * size of data vector
  * blocks number
  * threads per block number
  
  Collected data are presended in **Results** section.
  
## Code
  At the beggining, we declare size of vector, block and grid. These variables are point of this project.
  ```
  int nElem = 102173;
  dim3 block (15);
  dim3 grid ((nElem+block.x-1)/block.x);
  ```
  Secondly, we made pointers to vector in host and device and alloc memory.
  ``` 
   int *h_vect = (int *)malloc(nElem * sizeof(int));
   int *d_vect = NULL;
    
   cudaMalloc((void **)&d_vect, nElem * sizeof(int));
  ```
  Next, we set data to host memory and send it to device.
  ```
  cudaMemcpy(d_vect, h_vect, nElem * sizeof(int), cudaMemcpyHostToDevice);
  ```
  At the end we execute kernel function and send data back to host.
  ```
  vectorAdd<<<grid, block>>>(d_vect, nElem);
  cudaMemcpy(h_vect, d_vect, nElem * sizeof(int), cudaMemcpyDeviceToHost);

  ```
  
  
    

## Execution
  To compile this project we used command.
  ```
  nvcc grid_debug.cu -o grid_debug
  ```
  To measure execution time we used nvprof tool and we sent data to *.txt files.
  ```
  nvprof ./gird_debug 2>&1 | tee CM_VS_[SizeOfVector].txt
  ```
  sample result:
  ```
  ==2321== NVPROF is profiling process 2321, command: ./grid_debug
  grid.x 34 grid.y 1 grid.z 1
  block.x 3 block.y 1 block.z 1
  ==2321== Profiling application: ./grid_debug
  ==2321== Profiling result:
              Type  Time(%)      Time     Calls       Avg       Min       Max  Name
   GPU activities:   45.37%  1.5680us         1  1.5680us  1.5680us  1.5680us  vectorAdd(int*, int)
                     31.48%  1.0880us         1  1.0880us  1.0880us  1.0880us  [CUDA memcpy HtoD]
                     23.15%     800ns         1     800ns     800ns     800ns  [CUDA memcpy DtoH]
        API calls:   51.82%  321.39ms         1  321.39ms  321.39ms  321.39ms  cudaMalloc
                     34.41%  213.39ms         1  213.39ms  213.39ms  213.39ms  cudaFree
                     13.60%  84.340ms         1  84.340ms  84.340ms  84.340ms  cudaDeviceReset
                      0.08%  478.97us         1  478.97us  478.97us  478.97us  cuDeviceTotalMem
                      0.05%  335.66us        96  3.4960us     838ns  107.84us  cuDeviceGetAttribute
                      0.01%  74.101us         1  74.101us  74.101us  74.101us  cuDeviceGetName
                      0.01%  64.115us         1  64.115us  64.115us  64.115us  cudaLaunchKernel
                      0.01%  52.660us         2  26.330us  20.393us  32.267us  cudaMemcpy
                      0.00%  17.740us         2  8.8700us  1.7460us  15.994us  cuDeviceGet
                      0.00%  10.058us         1  10.058us  10.058us  10.058us  cuDeviceGetPCIBusId
                      0.00%  9.4990us         1  9.4990us  9.4990us  9.4990us  cudaDeviceSynchronize
                      0.00%  4.7490us         3  1.5830us     978ns  2.6540us  cuDeviceGetCount
                      0.00%  1.2570us         1  1.2570us  1.2570us  1.2570us  cuDeviceGetUuid

  ```
  then, we separated time data to [doc file](https://docs.google.com/spreadsheets/d/10RdRgu6PN2vl1llNBVsojPbfDENzBCP7uocNg2IaAhI/edit#gid=0) and made charts.


## Results

### 1. Dependence of the execution time on size of the vector
![alt text](https://github.com/AdriannaUrbanska/Introduction-to-CUDA-and-OpenCL/blob/master/CudaMalloc/Images/Dependence%20of%20the%20execution%20time%20on%20size%20of%20the%20vector.png)


### 2. Dependence of the execution time on blocks number
During the second experiment we were changing blocks number. The size of the vector was left to be constant and equal to 102173. We were setting values of block.x, block.y, block.z to be always equal.
![alt text](https://github.com/AdriannaUrbanska/Introduction-to-CUDA-and-OpenCL/blob/master/CudaMalloc/Images/Dependence%20of%20the%20execution%20time%20on%20blocks%20number.png)

We can observe on the plot that the shortest execution time for the vectorAdd function is when we set blocks number equals to 5.
There are no significant changes at execution time for CUDA memcpy HtoD and CUDA memcpy DtoH functions.

### 3. Dependence of the execution time on ThreadsPerBlock
In this point we changed our code and instead:
  ```
  dim3 block (15);
  dim3 grid ((nElem+block.x-1)/block.x);
  ```
we used:
```
  int threadsPerBlock = 8388608;
  int blocksPerGrid = (nElem+threadsPerBlock-1)/threadsPerBlock;
```
That modification is included in [grid_debug2.cu](https://github.com/AdriannaUrbanska/Introduction-to-CUDA-and-OpenCL/blob/master/CudaMalloc/src/grid_debug2.cu) file.
We were changing number of threadsPerBlock from 128 to 8388608. The size of the vector was left to be constant and equal to 102173.

![alt text](https://github.com/AdriannaUrbanska/Introduction-to-CUDA-and-OpenCL/blob/master/CudaMalloc/Images/Dependence%20of%20the%20execution%20time%20on%20ThreadsPerBlock.png)


![alt text](https://github.com/AdriannaUrbanska/Introduction-to-CUDA-and-OpenCL/blob/master/CudaMalloc/Images/Dependence%20of%20the%20excution%20time%20on%20ThreadsPerBlock%20for%20vectorAdd%20function.png)


## Authors

Adrianna Urbańska

Gabriel Chęć
