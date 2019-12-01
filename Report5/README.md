# Matrix Multiplication (NxM)

In this report our purpose is similar to previous one - [Report4](https://github.com/AdriannaUrbanska/Introduction-to-CUDA-and-OpenCL/tree/master/Report4) - we wrote a code with matrix multiplication algorithm using shared memory. One change is that our matrix does not have to be square this time.

## Code

Description of the code from the previous report is [here](https://github.com/AdriannaUrbanska/Introduction-to-CUDA-and-OpenCL/blob/master/Report4/README.md) in `Code` paragraph.

In this report's code [matrixMultiplication](https://github.com/AdriannaUrbanska/Introduction-to-CUDA-and-OpenCL/blob/master/Report5/src/matrixMultiplication.cu) the only change is that in `main()` function we created matrices which height and width are not equal.

```
Matrix a(N, M, M), g(M, N, N), ag(N, N, N);
```



## Authors

Adrianna Urbańska

Gabriel Chęć
