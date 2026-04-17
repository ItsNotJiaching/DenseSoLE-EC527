# DenseSoLE-EC527
To compile:

nvcc main.cu sole_serial.cpp sole_multi.cpp sole_gpu.cu -o test_sole

qrsh -l gpus=1 -l gpu_type=P100 -P ec527