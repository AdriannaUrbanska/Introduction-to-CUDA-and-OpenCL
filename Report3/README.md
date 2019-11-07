# CUDA Memory Management analysis

## First task

In the first part of this report we tested `cudaMallocManaged` function behavior in four different situations when Unified Memory is accessed:
* only by the GPU,
* only by the CPU,
* first by the GPU then the CPU,
* first by the CPU then the GPU.

To analyze our results we used NVIDIA Visual Profiler output. 


### 1.1 Access only by the GPU

![](https://github.com/AdriannaUrbanska/Introduction-to-CUDA-and-OpenCL/blob/master/Report3/img/onlyGPU_Img.png)
### 1.2. Access only by the CPU
![](https://github.com/AdriannaUrbanska/Introduction-to-CUDA-and-OpenCL/blob/master/Report3/img/onlyCPU_Img.png)
### 1.3. Access first by the GPU then the CPU
![](https://github.com/AdriannaUrbanska/Introduction-to-CUDA-and-OpenCL/blob/master/Report3/img/GPUthenCPU_Img.png)

### 1.4. Access first by the CPU then the GPU
![](https://github.com/AdriannaUrbanska/Introduction-to-CUDA-and-OpenCL/blob/master/Report3/img/CPUthenGPU_Img.png)

### 1.5. Summary

	                                 		Duration of cudaMallocManaged [ms]
           Access only by the CPU				308.34
           Access only by the GPU				339.69
           Access first by CPU then the GPU			339.94
           Access first by GPU then the CPU			325.19
	   
## Second part

In the second part we tested`cudaMemPrefetchAsync` function behavior in vector_add programs.

### 2.1 vector_add_standard

In this part we checked behaviour of the program when there was no `cudaMemPrefetchAsync` function ([vector_add_standard](https://github.com/AdriannaUrbanska/Introduction-to-CUDA-and-OpenCL/blob/master/Report3/src/vector_add_standard.cu) file). 
We allocated memory for vectors a, b, c using cudaMallocManaged function. Then we initialized values of the vectors using  CPU's initWith function. At the end we added vectors using kernel's addVectorsInto function.


Nvprof analysis:
```
           Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  154.35ms         1  154.35ms  154.35ms  154.35ms  addVectorsInto(float*, float*, float*, int)
      API calls:   65.43%  335.35ms         3  111.78ms  23.187us  335.28ms  cudaMallocManaged
                   30.12%  154.39ms         1  154.39ms  154.39ms  154.39ms  cudaDeviceSynchronize
                    4.26%  21.819ms         3  7.2730ms  6.8772ms  7.5657ms  cudaFree
                    0.09%  455.79us         1  455.79us  455.79us  455.79us  cuDeviceTotalMem
                    0.06%  331.96us        96  3.4570us     838ns  121.32us  cuDeviceGetAttribute
                    0.02%  115.03us         1  115.03us  115.03us  115.03us  cudaLaunchKernel
                    0.01%  46.305us         1  46.305us  46.305us  46.305us  cuDeviceGetName
                    0.00%  14.946us         1  14.946us  14.946us  14.946us  cudaGetDevice
                    0.00%  9.8480us         1  9.8480us  9.8480us  9.8480us  cuDeviceGetPCIBusId
                    0.00%  5.0280us         3  1.6760us     978ns  3.0030us  cuDeviceGetCount
                    0.00%  3.8410us         1  3.8410us  3.8410us  3.8410us  cudaDeviceGetAttribute
                    0.00%  2.7940us         2  1.3970us  1.0480us  1.7460us  cuDeviceGet
                    0.00%  1.2570us         1  1.2570us  1.2570us  1.2570us  cuDeviceGetUuid
                    0.00%     908ns         1     908ns     908ns     908ns  cudaGetLastError


```
### 2.2 vector_add_prefetch_gpu
Add new function ```cudaMemPrefetchAsync(a, size, deviceId);```

Nvprof:
```
           Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  2.5615ms         1  2.5615ms  2.5615ms  2.5615ms  addVectorsInto(float*, float*, float*, int)
      API calls:   79.82%  337.15ms         3  112.38ms  42.673us  337.00ms  cudaMallocManaged
                    9.92%  41.881ms         1  41.881ms  41.881ms  41.881ms  cudaDeviceSynchronize
                    6.76%  28.563ms         3  9.5211ms  9.0146ms  10.158ms  cudaFree
                    3.25%  13.728ms         3  4.5759ms  22.838us  13.554ms  cudaMemPrefetchAsync
                    0.12%  487.14us         1  487.14us  487.14us  487.14us  cuDeviceTotalMem
                    0.08%  338.17us        96  3.5220us     838ns  112.37us  cuDeviceGetAttribute
                    0.02%  105.11us         1  105.11us  105.11us  105.11us  cudaLaunchKernel
                    0.02%  72.006us         1  72.006us  72.006us  72.006us  cuDeviceGetName
                    0.00%  17.321us         1  17.321us  17.321us  17.321us  cudaGetDevice
                    0.00%  10.965us         1  10.965us  10.965us  10.965us  cuDeviceGetPCIBusId
                    0.00%  4.6090us         3  1.5360us     977ns  2.4450us  cuDeviceGetCount
                    0.00%  3.7710us         1  3.7710us  3.7710us  3.7710us  cudaDeviceGetAttribute
                    0.00%  2.5840us         2  1.2920us     978ns  1.6060us  cuDeviceGet
                    0.00%  1.2570us         1  1.2570us  1.2570us  1.2570us  cuDeviceGetUuid
                    0.00%     908ns         1     908ns     908ns     908ns  cudaGetLastError
```
### 2.3 vector_add_prefetch_gpu_init_gpu
Change CPU function ```initWith(3, a, N);``` into GPU kernel function ```initWith<<<numberOfBlocks, threadsPerBlock>>>(3, a, N);```
Nvprof:
```
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   50.77%  2.5666ms         1  2.5666ms  2.5666ms  2.5666ms  addVectorsInto(float*, float*, float*, int)
                   49.23%  2.4890ms         3  829.66us  828.07us  831.08us  initWith(float, float*, int)
      API calls:   91.10%  302.58ms         3  100.86ms  22.838us  302.51ms  cudaMallocManaged
                    6.21%  20.616ms         3  6.8719ms  4.7836ms  11.005ms  cudaFree
                    1.51%  5.0278ms         1  5.0278ms  5.0278ms  5.0278ms  cudaDeviceSynchronize
                    0.87%  2.9002ms         3  966.72us  946.35us  996.29us  cudaMemPrefetchAsync
                    0.14%  450.83us         1  450.83us  450.83us  450.83us  cuDeviceTotalMem
                    0.11%  359.61us        96  3.7450us     838ns  137.66us  cuDeviceGetAttribute
                    0.03%  100.36us         4  25.090us  8.4510us  70.120us  cudaLaunchKernel
                    0.02%  52.521us         1  52.521us  52.521us  52.521us  cuDeviceGetName
                    0.00%  15.156us         1  15.156us  15.156us  15.156us  cudaGetDevice
                    0.00%  10.756us         1  10.756us  10.756us  10.756us  cuDeviceGetPCIBusId
                    0.00%  6.0050us         3  2.0010us     977ns  3.8410us  cuDeviceGetCount
                    0.00%  2.7940us         2  1.3970us     978ns  1.8160us  cuDeviceGet
                    0.00%  1.5360us         1  1.5360us  1.5360us  1.5360us  cudaDeviceGetAttribute
                    0.00%  1.1180us         1  1.1180us  1.1180us  1.1180us  cuDeviceGetUuid
                    0.00%     838ns         1     838ns     838ns     838ns  cudaGetLastError

```
### 2.4 vector_add_prefetch_gpucpu_init_gpu.cu
Add ```cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);``` function.
Nvprof:
```
GPU activities:   50.82%  2.5691ms         1  2.5691ms  2.5691ms  2.5691ms  addVectorsInto(float*, float*, float*, int)
                   49.18%  2.4862ms         3  828.72us  826.85us  830.02us  initWith(float, float*, int)
      API calls:   81.88%  337.30ms         3  112.43ms  22.978us  337.23ms  cudaMallocManaged
                   11.29%  46.522ms         4  11.631ms  953.26us  43.600ms  cudaMemPrefetchAsync
                    5.36%  22.094ms         3  7.3646ms  4.8523ms  12.241ms  cudaFree
                    1.22%  5.0273ms         1  5.0273ms  5.0273ms  5.0273ms  cudaDeviceSynchronize
                    0.11%  453.97us         1  453.97us  453.97us  453.97us  cuDeviceTotalMem
                    0.08%  333.49us        96  3.4730us     838ns  110.98us  cuDeviceGetAttribute
                    0.02%  101.13us         4  25.282us  9.0100us  70.749us  cudaLaunchKernel
                    0.02%  69.562us         1  69.562us  69.562us  69.562us  cuDeviceGetName
                    0.00%  16.134us         1  16.134us  16.134us  16.134us  cudaGetDevice
                    0.00%  10.546us         1  10.546us  10.546us  10.546us  cuDeviceGetPCIBusId
                    0.00%  4.4700us         3  1.4900us     908ns  2.4450us  cuDeviceGetCount
                    0.00%  3.0030us         2  1.5010us  1.3970us  1.6060us  cuDeviceGet
                    0.00%  1.6760us         1  1.6760us  1.6760us  1.6760us  cudaDeviceGetAttribute
                    0.00%  1.1870us         1  1.1870us  1.1870us  1.1870us  cuDeviceGetUuid
                    0.00%     908ns         1     908ns     908ns     908ns  cudaGetLastError


```
### 2.5 Summary

						time of addVectorsInto [ms]	time of cudaMallocManaged [ms]
	vector_add_standard				154.35				335.35
	vector_add_prefetch_gpu				2.56				337.15
	vector_add_prefetch_gpu_init_gpu		2.57				302.58
	vector_add_prefetch_cpugpu_init_gpu		2.57				337.30


## Authors

Adrianna Urbańska

Gabriel Chęć
