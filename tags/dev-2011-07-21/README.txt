To use Visual Studio you should install NVIDA's Parallel Insight product.
http://parallelnsight.nvidia.com/

Right click on your project, Properties, VC++ Directories:
Add $(CUDA_INC_PATH); to Include Directories.
Add $(CUDA_LIB_PATH); to Library Directories.

Add cuda.lib;cudart.lib; to Linker, Input, Additional Dependencies.
