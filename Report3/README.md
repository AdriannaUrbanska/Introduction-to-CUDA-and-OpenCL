# CUDA Memory Management analysis

In this report we tested `cudaMallocManaged` function behavior in four different situations when Unified Memory is accessed:
* only by the GPU,
* only by the CPU,
* first by the GPU then the CPU,
* first by the CPU then the GPU.

To analyze our results we used NVIDIA Visual Profiler output. 


### 1. Access only by the GPU

![](https://github.com/AdriannaUrbanska/Introduction-to-CUDA-and-OpenCL/blob/master/Report3/img/onlyGPU_Img.png)
### 2. Access only by the CPU
![](https://github.com/AdriannaUrbanska/Introduction-to-CUDA-and-OpenCL/blob/master/Report3/img/onlyCPU_Img.png)
### 3. Access first by the GPU then the CPU
![](https://github.com/AdriannaUrbanska/Introduction-to-CUDA-and-OpenCL/blob/master/Report3/img/GPUthenCPU_Img.png)

### 4. Access first by the CPU then the GPU
![](https://github.com/AdriannaUrbanska/Introduction-to-CUDA-and-OpenCL/blob/master/Report3/img/CPUthenGPU_Img.png)

## Authors

Adrianna Urbańska

Gabriel Chęć
