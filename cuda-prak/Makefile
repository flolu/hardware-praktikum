ARCH= -gencode arch=compute_35,code=sm_35 -gencode arch=compute_75,code=sm_75

all:
	nvcc ${ARCH} -o cuda-prak main.cu -lpng

perf:
	nvcc -O3 ${ARCH} -o cuda-perf perf.cu -lpng 

perf-int:
	nvcc -O3 ${ARCH} -o cuda-perf-int perf.cu -lpng -DUSEINT

int:
	nvcc ${ARCH} -o cuda-prak-int main.cu -lpng -DUSEINT

debug:
	nvcc -G -g ${ARCH} -o cuda-debug main.cu -lpng

debug-int:
	nvcc -G -g ${ARCH} -o cuda-debug-int main.cu -lpng -DUSEINT
