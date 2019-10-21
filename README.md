# CUDA functions execution dependencies

  In this report we research dependencies of execution time of functions in CUDA library. 
  We measured execution time according to:
  * size of data vector
  * threds per block
  
  Collected data are presended in *Results* section.
  
    ---
## Documentation
To measure execution time we used nvprof tool and we send data to *.txt files.
### Example
```
nvprof ./gird_debug 2>&1 | tee CM_VS100.txt
```


## Results

## Authors

Adrianna Urbańska
Gabriel Chęć
