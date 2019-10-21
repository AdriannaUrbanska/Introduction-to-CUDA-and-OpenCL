# CUDA functions execution dependencies

  In this report we research dependencies of execution time of functions in CUDA library. 
  We measured execution time according to:
  * size of data vector
  * threds per block
  
  Collected data are presended in *Results* section.
  
## Execution

To measure execution time we used nvprof tool and we send data to *.txt files.
```
nvprof ./gird_debug 2>&1 | tee CM_VS100.txt
```
sample results:
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


## Results

## Authors

Adrianna Urbańska
Gabriel Chęć
