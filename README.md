# DenseSoLE-EC527
### Jiaxing Wang, Alvin Yan, Owen Jiang

### Summary
This project explores the implementation and optimization of dense systems of linear equations.

### Compilation Notes
To compile:

nvcc -Xcompiler -fopenmp main.cu sole_serial.cpp sole_multi.cpp sole_gpu.cu -o test_sole

To request a GPU on the SCC:

qrsh -l gpus=1 -l gpu_type=P100 -P ec527

This will request a 6.0 compute GPU.