main.out poc.out: %.out: %.cc
	clang++ $^ -std=c++11 -march=cascadelake -o $@ -O3

clean:
	rm -f *.out *.ll main poc
