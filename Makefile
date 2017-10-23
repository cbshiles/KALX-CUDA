CUDA_PATH = /usr/local/pkg/cuda/3.0/cuda
CXX = g++
CXXFLAGS = -fPIC -g -Wall
LIBS = -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib -lcuda
INCS = -I$(CUDA_PATH)/include -I..

NVCC = nvcc
NVCCFLAGS = --cubin --compiler-options -fno-strict-aliasing $(INCS) -DUNIX -g

%.cubin: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

empty: empty.cpp empty.cubin
	$(CXX) $(CXXFLAGS) $(INCS) -o $@ $(LIBS) $<

clean:
	rm -f empty *.o *.cubin
