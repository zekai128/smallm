NVCC     = nvcc
CFLAGS   = -std=c++17
LIBS     = -lcublas

all: main

main: main.cu tensor.cu
	$(NVCC) $(CFLAGS) $(LIBS) -o main main.cu tensor.cu

clean:
	rm -f main
