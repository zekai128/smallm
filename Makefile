NVCC    = nvcc
CFLAGS  = -std=c++17 -I. -dc -g
LIBS    = -lcublas

KERNEL_SRCS = $(wildcard kernels/*.cu)
KERNEL_OBJS = $(KERNEL_SRCS:.cu=.o)

COMMON_OBJS = tensor.o transformer.o $(KERNEL_OBJS)

all: main training inference

main: main.o $(COMMON_OBJS)
	$(NVCC) $(LIBS) -o $@ $^

training: training.o $(COMMON_OBJS)
	$(NVCC) $(LIBS) -o $@ $^

inference: inference.o $(COMMON_OBJS)
	$(NVCC) $(LIBS) -o $@ $^

%.o: %.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f main training *.o kernels/*.o
