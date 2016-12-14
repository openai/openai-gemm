lib/c_interface.o: gen_kernels.py src/c_interface.cu include/kernel_headers.h include/static_kernel_information.h
	python gen_kernels.py
	nvcc -c src/c_interface.cu -o lib/c_interface.o -std=c++11 -I include/ -I . 

test: src/test.cu
	nvcc -o test src/test.cu lib/c_interface.o -std=c++11 -I include -lcuda
