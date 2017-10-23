// cu.h - CUDA driver API
// Copyright (c) 2011 KALX, LLC. All rights reserved. No warranty is made.
#pragma once
#include "ensure.h"
#include <cuda.h>

#ifndef ensure
#define ensure(x) assert(x)
#endif 

namespace cu {

	// cudaError_enum - cudaGetErrorString???
	inline const char* Result(int result)
	{
		switch (result) {
			case CUDA_SUCCESS: 
				return "CUDA_SUCCESS";
			case CUDA_ERROR_INVALID_VALUE: 
				return "CUDA_ERROR_INVALID_VALUE";
			case CUDA_ERROR_OUT_OF_MEMORY: 
				return "CUDA_ERROR_OUT_OF_MEMORY";
			case CUDA_ERROR_NOT_INITIALIZED: 
				return "CUDA_ERROR_NOT_INITIALIZED";
			case CUDA_ERROR_DEINITIALIZED: 
				return "CUDA_ERROR_DEINITIALIZED";
			case CUDA_ERROR_PROFILER_DISABLED: 
				return "CUDA_ERROR_PROFILER_DISABLED";
			case CUDA_ERROR_PROFILER_NOT_INITIALIZED: 
				return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
			case CUDA_ERROR_PROFILER_ALREADY_STARTED: 
				return "CUDA_ERROR_PROFILER_ALREADY_STARTED";
			case CUDA_ERROR_PROFILER_ALREADY_STOPPED: 
				return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
			case CUDA_ERROR_NO_DEVICE: 
				return "CUDA_ERROR_NO_DEVICE";
			case CUDA_ERROR_INVALID_DEVICE: 
				return "CUDA_ERROR_INVALID_DEVICE";
			case CUDA_ERROR_INVALID_IMAGE: 
				return "CUDA_ERROR_INVALID_IMAGE";
			case CUDA_ERROR_INVALID_CONTEXT: 
				return "CUDA_ERROR_INVALID_CONTEXT";
			case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: 
				return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
			case CUDA_ERROR_MAP_FAILED: 
				return "CUDA_ERROR_MAP_FAILED";
			case CUDA_ERROR_UNMAP_FAILED: 
				return "CUDA_ERROR_UNMAP_FAILED";
			case CUDA_ERROR_ARRAY_IS_MAPPED: 
				return "CUDA_ERROR_ARRAY_IS_MAPPED";
			case CUDA_ERROR_ALREADY_MAPPED: 
				return "CUDA_ERROR_ALREADY_MAPPED";
			case CUDA_ERROR_NO_BINARY_FOR_GPU: 
				return "CUDA_ERROR_NO_BINARY_FOR_GPU";
			case CUDA_ERROR_ALREADY_ACQUIRED: 
				return "CUDA_ERROR_ALREADY_ACQUIRED";
			case CUDA_ERROR_NOT_MAPPED: 
				return "CUDA_ERROR_NOT_MAPPED";
			case CUDA_ERROR_NOT_MAPPED_AS_ARRAY: 
				return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
			case CUDA_ERROR_NOT_MAPPED_AS_POINTER: 
				return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
			case CUDA_ERROR_ECC_UNCORRECTABLE: 
				return "CUDA_ERROR_ECC_UNCORRECTABLE";
			case CUDA_ERROR_UNSUPPORTED_LIMIT: 
				return "CUDA_ERROR_UNSUPPORTED_LIMIT";
			case CUDA_ERROR_CONTEXT_ALREADY_IN_USE: 
				return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
			case CUDA_ERROR_INVALID_SOURCE: 
				return "CUDA_ERROR_INVALID_SOURCE";
			case CUDA_ERROR_FILE_NOT_FOUND: 
				return "CUDA_ERROR_FILE_NOT_FOUND";
			case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: 
				return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
			case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED: 
				return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
			case CUDA_ERROR_OPERATING_SYSTEM: 
				return "CUDA_ERROR_OPERATING_SYSTEM";
			case CUDA_ERROR_INVALID_HANDLE: 
				return "CUDA_ERROR_INVALID_HANDLE";
			case CUDA_ERROR_NOT_FOUND: 
				return "CUDA_ERROR_NOT_FOUND";
			case CUDA_ERROR_NOT_READY: 
				return "CUDA_ERROR_NOT_READY";
			case CUDA_ERROR_LAUNCH_FAILED: 
				return "CUDA_ERROR_LAUNCH_FAILED";
			case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: 
				return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
			case CUDA_ERROR_LAUNCH_TIMEOUT: 
				return "CUDA_ERROR_LAUNCH_TIMEOUT";
			case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: 
				return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
			case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: 
				return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
			case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: 
				return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
			case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: 
				return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
			case CUDA_ERROR_CONTEXT_IS_DESTROYED: 
				return "CUDA_ERROR_CONTEXT_IS_DESTROYED";

			default: 
				return "CUDA_ERROR_UNKNOWN";
		}
	}

	// need exactly one of these
	struct Init {
		Init()
		{
			ensure (CUDA_SUCCESS == cuInit(0));
		}
	};

	class Device {
		CUdevice dev_;
		Device(const Device&);
		Device& operator=(const Device&);
	public:
		Device(int ordinal = 0)
		{
			// calls cuInit if necesarry
			ensure (ordinal < Count());

			CUresult result = cuDeviceGet(&dev_, ordinal);
			ensure (CUDA_SUCCESS == result);
		}
		~Device()
		{
		}
		operator CUdevice()
		{
			return dev_;
		}
		void* operator&()
		{
			return &dev_;
		}
		static int Count(void)
		{
			static Init i_;
			int count;

			CUresult result = cuDeviceGetCount(&count);
			ensure (CUDA_SUCCESS == result);

			return count;
		}
	};

	// context - CUDA process
	class Ctx {
		CUcontext ctx_;
		Ctx(const Ctx&);
		Ctx& operator=(const Ctx&);
	public:
		Ctx(unsigned int flags = 0, CUdevice device = 0)
		{
			CUresult result = cuCtxCreate(&ctx_, flags, device);
			ensure (CUDA_SUCCESS == result);
		}
		~Ctx()
		{
			CUresult result = cuCtxDestroy(ctx_);
			ensure (CUDA_SUCCESS == result);
		}
		operator CUcontext()
		{
			return ctx_;
		}
	};

	// cubin file
	class Module {
		CUmodule mod_;
		Module(const Module&);
		Module& operator=(Module&);
	public:
		struct Load 
		{
			const char* file_;
			Load(const char* file)
				: file_(file)
			{ }
		};
		struct LoadData 
		{
			const char* data_;
			LoadData(const char* data)
				: data_(data)
			{ }
		};
/*		class LoadDataEx 
		{
			const void* image_;
			std::vector<CUjit_option> option_;
			mutable std::vector<void*> value_;
			union datum {
				unsigned int ui;
				float f;
				char* s;
			};
			std::vector<datum> datum_;
			float t_; // wall time
		public:
			LoadDataEx(const void* image)
				: image_(image), t_(0)
			{ }
			LoadDataEx& option(CUjit_option o, unsigned int ui = 0)
			{
				option_.push_back(o);
				datum d;
				d.ui = ui;
				datum_.push_back(d);

				return *this;
			}
			LoadDataEx& option(CUjit_option o, char* s)
			{
				option_.push_back(o);
				datum d;
				d.s = s;
				datum_.push_back(d);

				return *this;
			}
			LoadDataEx& maxRegisters(unsigned int ui)
			{
				return option(CU_JIT_MAX_REGISTERS, ui);
			}
			LoadDataEx& threadsPerBlock(unsigned int ui)
			{
				return option(CU_JIT_THREADS_PER_BLOCK, ui);
			}
			LoadDataEx& wallTime()
			{
				option_.push_back(CU_JIT_WALL_TIME);
				datum d;
				d.f = t_;
				datum_.push_back(d);

				return *this;
			}
			float wallTime() const
			{
				return t_;
			}
			LoadDataEx& infoLogBuffer(char* s)
			{
				return option(CU_JIT_INFO_LOG_BUFFER, s);
		}
			LoadDataEx& infoLogBufferSize(unsigned int ui)
			{
				return option(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, ui);
			}
			LoadDataEx& errorLogBuffer(char* s)
			{
				return option(CU_JIT_ERROR_LOG_BUFFER, s);
			}
			LoadDataEx& errorLogBufferSize(unsigned int ui)
			{
				return option(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, ui);
			}
			LoadDataEx& jitOptimizationLevel(unsigned int ui)
			{
				ensure (0 <= ui && ui <= 4);
				return option(CU_JIT_OPTIMIZATION_LEVEL, ui);
			}
			LoadDataEx& jitTargetFromCucontext()
			{
				return option(CU_JIT_TARGET_FROM_CUCONTEXT);
			}
			LoadDataEx& jitTarget(unsigned int ui)
			{
				return ui < 10 ? option(CU_JIT_TARGET, ui)
				    : ui == 10 ? option(CU_JIT_TARGET, CU_TARGET_COMPUTE_10)
				    : ui == 11 ? option(CU_JIT_TARGET, CU_TARGET_COMPUTE_11)
				    : ui == 12 ? option(CU_JIT_TARGET, CU_TARGET_COMPUTE_12)
				    : ui == 13 ? option(CU_JIT_TARGET, CU_TARGET_COMPUTE_13)
				    : ui == 20 ? option(CU_JIT_TARGET, CU_TARGET_COMPUTE_20)
				    : throw std::runtime_error("unknown jit target"), *this;
			}
			LoadDataEx& preferPtx(void)
			{
				return option(CU_JIT_FALLBACK_STRATEGY, CU_PREFER_PTX);
			}
			LoadDataEx& preferBinary(void)
			{
				return option(CU_JIT_FALLBACK_STRATEGY, CU_PREFER_BINARY);
			}

			// for cuModuleLoadDataEx
			const void* image(void) const
			{
				return image_;
			}
			unsigned int numOptions(void) const
			{
				return static_cast<unsigned int>(option_.size());
			}
			CUjit_option* options(void) const
			{
				return const_cast<CUjit_option*>(&option_[0]);
			}
			void** optionValues(void) const
			{
				for (size_t i = 0; i < datum_.size(); ++i)
					value_.push_back((void*)&datum_[i]);

				return const_cast<void**>(&value_[0]);
			}
		};
*/
		Module(const Load& file)
		{
			CUresult result = cuModuleLoad(&mod_, file.file_);
			ensure (CUDA_SUCCESS == result);
		}
		Module(const LoadData& data)
		{
			CUresult result = cuModuleLoadData(&mod_, data.data_);
			ensure (CUDA_SUCCESS == result);
		}
		/*
		Module(const LoadDataEx& image)
		{
			CUresult result = cuModuleLoadDataEx(&mod_, image.image(), image.numOptions(), image.options(), image.optionValues());
			ensure (CUDA_SUCCESS == result);
		}
		*/
		~Module()
		{
			CUresult result = cuModuleUnload(mod_);
			ensure (CUDA_SUCCESS == result);
		}
		CUfunction GetFunction(const char* name)
		{
			CUfunction f;

			CUresult result = cuModuleGetFunction(&f, mod_, name);
			ensure (CUDA_SUCCESS == result);

			return f;
		}
	};

	class Stream {
		CUstream str_;
		Stream(const Stream&);
		Stream& operator=(const Stream&);
	public:
		Stream(unsigned int flags = 0)
		{
			CUresult result = cuStreamCreate(&str_, flags);
			ensure (CUDA_SUCCESS == result);
		}
		~Stream()
		{
			CUresult result = cuStreamDestroy(str_);
			ensure (CUDA_SUCCESS == result);
		}
		operator CUstream()
		{
			return str_;
		}
	};

	class Event {
		CUevent event_;
		Event(const Event&);
		Event& operator=(const Event&);
	public:
		Event(unsigned int flags = CU_EVENT_DEFAULT)
		{
			CUresult result = cuEventCreate(&event_, flags);
			ensure (CUDA_SUCCESS == result);
		}
		~Event()
		{
			CUresult result = cuEventDestroy(event_);
			ensure (CUDA_SUCCESS == result);
		}
		operator CUevent()
		{
			return event_;
		}
	};
	inline unsigned int align(unsigned int o, unsigned int a) { return (o + a - 1)&~(a - 1); }
	class Launch {
		CUfunction func_;
		int width_, height_;
		CUstream str_;
		unsigned int off_;
		Launch(const Launch&);
		Launch& operator=(const Launch&);
	public:
		// usage:
		// CUfunction f = Module(Module::Load("file.cubin")).GetFunction("function");
		// Launch(f, w?, h?, s?)
		// .BlockShape(x, y)
		// .SetSharedSize(size)
		// .ParamSet(1u).ParamSet(1.2f)...
		Launch(CUfunction func, int width = 0, int height = 1, CUstream str = 0)
			: func_(func), width_(width), height_(height), str_(str), off_(0)
		{
		}	
		~Launch()
		{
			CUresult result;

			result = cuParamSetSize(func_, off_);
			ensure (CUDA_SUCCESS == result);

			if (width_ == 0 || height_ == 0)
				result = cuLaunch(func_);
			else if (str_ == 0)
				result = cuLaunchGrid(func_, width_, height_);
			else
				result = cuLaunchGridAsync(func_, width_, height_, str_);

			ensure (CUDA_SUCCESS == result);
		}
		Launch& SetBlockShape(int x, int y = 1, int z = 1)
		{
			CUresult result;
			
			result = cuFuncSetBlockShape(func_, x, y, z);
			ensure (CUDA_SUCCESS == result);

			return *this;	
		}
		Launch& SetSharedSize(unsigned int bytes)
		{
			CUresult result;
			
			result = cuFuncSetSharedSize(func_, bytes);
			ensure (CUDA_SUCCESS == result);

			return *this;	
		}
		Launch& ParamSet(void* ptr)
		{
			off_ = align(off_, __alignof(ptr));
			CUresult result = cuParamSetv(func_, off_, ptr, sizeof(ptr));
			ensure (CUDA_SUCCESS == result);
			off_ += sizeof(ptr);

			return *this;
		}
		Launch& ParamSet(unsigned int i)
		{
			off_ = align(off_, __alignof(i));
			CUresult result = cuParamSeti(func_, off_, i);
			ensure (CUDA_SUCCESS == result);
			off_ += sizeof(unsigned int);

			return *this;
		}
		Launch& ParamSet(float f)
		{
			off_ = align(off_, __alignof(f));
			CUresult result = cuParamSetf(func_, off_, f);
			ensure (CUDA_SUCCESS == result);
			off_ += sizeof(float);

			return *this;
		}
	};
/*
	#define ALIGN_UP(o, a) o = (o + a - 1)&~(a - 1)

	int o = 0;
	void* ptr = (void*)(size_t)A;
	ALIGN_UP(o, __alignof(ptr));
	cuParamSetv(f, o, ptr, sizeof(ptr));

	o += sizeof(ptr);
	...
	cuParamSetSize(f, o);

	template<int x,int y,int z, unsigned int size>
	CUresult Launch(CUfunction f)
	{
		cuFuncSetBlockShape(f, x, y, z);
		cuFuncSetSharedSize(f, size);

		return cuLaunch(f);
	}
*/
	namespace Mem {

		template<class T>
		class Host {
			T* mem_;
			unsigned int size_;
			Host(const Host&);
			Host& operator=(const Host&);
		public:
			// page locked host memory
			Host(unsigned int size, unsigned int flags = 0)
				: size_(size)
			{
				CUresult result = cuMemHostAlloc((void**)&mem_, size_*sizeof(T), flags);
				ensure (CUDA_SUCCESS == result);
			}
			~Host()
			{
				cuMemFreeHost(mem_);
			}
			operator T*()
			{
				return mem_;
			}
			operator const T*() const
			{
				return mem_;
			}
			unsigned int Size() const
			{
				return size_*sizeof(T);
			}
			T& operator[](unsigned int i)
			{
				// check bounds
				return mem_[i];
			}
			const T& operator[](unsigned int i) const
			{
				// check bounds
				return mem_[i];
			}
		};

		template<class T = unsigned char>
		class Device {
			CUdeviceptr ptr_;
			unsigned int size_;
			Device(const Device&);
			Device& operator=(const Device&);
		public:
			Device(unsigned int size)
				: size_(size)
			{
				CUresult result = cuMemAlloc(&ptr_, size*sizeof(T));
				ensure (CUDA_SUCCESS == result);
			}
			~Device()
			{
				cuMemFree(ptr_);
			}
			operator CUdeviceptr()
			{
				return ptr_;
			}
			operator T*()
			{
				return (T*)&ptr_;
			}
			operator const T*() const
			{
				return (const T*)&ptr_;
			}
/*			void* operator&()
			{
				return &ptr_;
			}
*//*			operator T*()
			{
				return ptr_;
			}
*/
/*
			const operator CUdeviceptr() const
			{
				return ptr_;
			}
*/			unsigned int Size() const
			{
				return size_*sizeof(T);
			}
			/*
			T& operator[](size_t i)
			{
				// check bounds
				return ptr_[i*sizeof(T)];
			}
			const T& operator[](size_t i) const
			{
				// check bounds
				return ptr_[i*sizeof(T)];
			}
			*/
		};

		template<class T>
		struct traits { static const CUarray_format format; };
		template<> struct traits<unsigned char> 
		{ static const CUarray_format format = CU_AD_FORMAT_UNSIGNED_INT8; };
		template<> struct traits<unsigned short> 
		{ static const CUarray_format format = CU_AD_FORMAT_UNSIGNED_INT16; };
		template<> struct traits<unsigned int> 
		{ static const CUarray_format format = CU_AD_FORMAT_UNSIGNED_INT32; };
		template<> struct traits<char> 
		{ static const CUarray_format format = CU_AD_FORMAT_SIGNED_INT8; };
		template<> struct traits<short> 
		{ static const CUarray_format format = CU_AD_FORMAT_SIGNED_INT16; };
		template<> struct traits<int> 
		{ static const CUarray_format format = CU_AD_FORMAT_SIGNED_INT32; };
//		template<float16_t> traits<> struct { CUarray_format format = CU_AD_FORMAT_HALF; };
		template<> struct traits<float> 
		{ static const CUarray_format format = CU_AD_FORMAT_FLOAT; };

		template<class T>
		class Array {
			CUarray ptr_;
			CUDA_ARRAY_DESCRIPTOR cad_;
			Array(const Array&);
			Array& operator=(const Array&);
		public:
			Array(unsigned int width, unsigned height, unsigned int channels = 1)
			{
				cad_.Width = width;
				cad_.Height = height;
				cad_.Format = Mem::traits<T>::format;
				cad_.NumChannels = channels;

				CUresult result = cuArrayCreate(&ptr_, &cad_);
				ensure (CUDA_SUCCESS == result);
			}
			~Array()
			{
				cuArrayDestroy(ptr_);
			}
			operator CUarray()
			{
				return ptr_;
			}
			unsigned int Width(void) const
			{
				return cad_.Width;
			}
			unsigned int Height(void) const
			{
				return cad_.Height;
			}
			unsigned int NumChannels(void) const
			{
				return cad_.NumChannels;
			}
			CUarray_format Format(void) const
			{
				return cad_.Format;
			}
			unsigned int Size(void) const
			{
				return Width()*Height()*NumChannels()*sizeof(T);
			}
			// T operator(...)
		};
		template<class T>
		class Array3D {
			CUarray ptr_;
			CUDA_ARRAY3D_DESCRIPTOR cad_;
			Array3D(const Array3D&);
			Array3D& operator=(const Array3D&);
		public:
			Array3D(unsigned int width, unsigned height, unsigned int depth, unsigned int channels = 1)
			{
				cad_.Width = width;
				cad_.Height = height;
				cad_.Depth = depth;
				cad_.Format = Mem::traits<T>::format;
				cad_.NumChannels = channels;

				CUresult result = cuArray3DCreate(&ptr_, &cad_);
				ensure (CUDA_SUCCESS == result);
			}
			~Array3D()
			{
				cuArrayDestroy(ptr_);
			}
			operator CUarray()
			{
				return ptr_;
			}
			unsigned int Width(void) const
			{
				return cad_.Width;
			}
			unsigned int Height(void) const
			{
				return cad_.Height;
			}
			unsigned int Depth(void) const
			{
				return cad_.Depth;
			}
			unsigned int NumChannels(void) const
			{
				return cad_.NumChannels;
			}
			CUarray_format Format(void) const
			{
				return cad_.Format;
			}
			unsigned int Size(void) const
			{
				return Width()*Height()*NumChannels()*sizeof(T);
			}
		};
	} // namespace Mem

	class TexRef {
		CUtexref tex_;
		TexRef(const TexRef&);
		TexRef& operator=(const TexRef&);
	public:
		TexRef()
		{
			CUresult result = cuTexRefCreate(&tex_);
			ensure (CUDA_SUCCESS == result);
		}
		~TexRef()
		{
			CUresult result = cuTexRefDestroy(tex_);
			ensure (CUDA_SUCCESS == result);
		}
		operator CUtexref()
		{
			return tex_;
		}
	};

	template<class T>
	const Mem::Host<T>& operator>>(const Mem::Host<T>& h, Mem::Device<T>& d)
	{
		ensure (h.Size() <= d.Size());
		CUresult result = cuMemcpyHtoD(d, h, h.Size());
		ensure (CUDA_SUCCESS == result);

		return h;
	}
	template<class T>
	const Mem::Host<T>& operator>>(const Mem::Host<T>& h, Mem::Array<T>& a)
	{
		ensure (h.Size() <= a.Size());
		CUresult result = cuMemcpyHtoA(a, 0, h, h.Size());
		ensure (CUDA_SUCCESS == result);

		return h;
	}
	// etc...
	template<class T>
	Mem::Host<T>& operator<<(Mem::Host<T>& h, /*const*/ Mem::Device<T>& d)
	{
		ensure (h.Size() >= d.Size());
		CUresult result = cuMemcpyDtoH(h, d, d.Size()); // const_cast<CUdeviceptr>(d)
		ensure (CUDA_SUCCESS == result);

		return h;
	}
	template<class T>
	Mem::Host<T>& operator<<(Mem::Host<T>& h, const Mem::Array<T>& a)
	{
		ensure (h.Size() >= a.Size());
		CUresult result = cuMemcpyAtoH(h, a, 0, a.Size());
		ensure (CUDA_SUCCESS == result);

		return h;
	}
	// etc...
} // namespace cu
