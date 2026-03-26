NVCC     = nvcc
CFLAGS   = -std=c++17 -I.
LIBS     = -lcublas

SRCS     = main.cu tensor.cu $(wildcard kernels/*.cu)

all: main

main: $(SRCS)
	$(NVCC) $(CFLAGS) $(LIBS) -o main $(SRCS)

clean:
	rm -f main
