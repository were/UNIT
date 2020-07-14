micro.out main.out poc.out: %.out: %.cc
	clang++ $^ -std=c++11 -march=cascadelake -o $@ -O3

poc.exe gemm.exe: %.exe: %.cu
	nvcc -arch=sm_70 $^ -O2 -o $@

mkldnn_conv.out mkldnn_gemm.out: %.out: %.cc
	clang++ $^ -std=c++11 -march=cascadelake -o $@ -O3 \
	  -I../../../oneDNN/include -I../../../oneDNN/build/include \
	  -L../../../oneDNN/build/src -ldnnl -lm -lpthread -lz

clean:
	rm -f *.out *.ll main poc *.exe *.s *.o
