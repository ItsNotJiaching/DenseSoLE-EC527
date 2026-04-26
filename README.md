# DenseSoLE-EC527
### Jiaxing Wang, Alvin Yan, Owen Jiang

### Summary
This project explores the implementation and optimization of dense systems of linear equations.

### Compilation Notes
To compile:

nvcc -O1 -Xcompiler -fopenmp main.cu sole_serial.cpp sole_multi.cpp sole_gpu.cu -o test_sole -lcusolver -lcublas -lm

To Run:
OMP_NUM_THREADS=8,3 ./test_sole

To request a GPU on the SCC:

qrsh -l gpus=1 -l gpu_type=P100 -P ec527

This will request a 6.0 compute GPU.

Make sure to do "module load cuda" to get access to the nvcc compiler.