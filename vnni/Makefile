micro.out main.out poc.out: %.out: %.cc
	clang++ $^ -std=c++11 -march=cascadelake -o $@ -O3

poc.exe gemm.exe: %.exe: %.cu
	nvcc -arch=sm_70 $^ -O2 -o $@

mkldnn_conv.out mkldnn_gemm.out: %.out: %.cc
	clang++ $^ -std=c++11 -march=cascadelake -o $@ -O3 \
	  -I../../mkl-dnn/include -I../../mkl-dnn/build/include \
	  -L../../mkl-dnn/build/src -lmkldnn -lm -lpthread -lz

clean:
	rm -f *.out *.ll main poc *.exe *.s *.o
