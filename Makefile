main.out poc.out: %.out: %.cc
	clang++ $^ -std=c++11 -march=cascadelake -o $@ -O3

poc.exe gemm.exe: %.exe: %.cu
	nvcc -arch=sm_70 $^ -O2 -o $@

clean:
	rm -f *.out *.ll main poc *.exe
