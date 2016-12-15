gen_kernels: gen_kernels.py include/kernel_headers.h $(wildcard sass/*.sass)
	python gen_kernels.py

lib/c_interface.o: gen_kernels src/c_interface.cpp include/kernel_headers.h include/static_kernel_information.h
	nvcc -c src/c_interface.cpp -o lib/c_interface.o -std=c++11 -I .

test: src/test.cu lib/c_interface.o
	nvcc -o test src/test.cu lib/c_interface.o -std=c++11 -I . -lcuda

clean:
	rm -f include/kernels/*
	rm -rf temp/
	rm -f lib/c_interface.o
	rm -f test
