// cu.h - CUDA driver API
// Copyright (c) 2011 KALX, LLC. All rights reserved. No warranty is made.
#pragma once
#include "ensure.h"
#include <cuda.h>

namespace cu {

	// need exactly one of these
	struct Init {
		static CUresult result()
		{
			static CUresult result_(cuInit(0));

			return result_;
		}
		Init()
		{	
			ensure (CUDA_SUCCESS == result());
		}
	};

	// device handle from cuDeviceGet
	class Device {
		CUdevice dev_;
		Device(const Device&);
		Device& operator=(const Device&);
	public:
		Device(int ordinal = 0)
		{
			int count;

			ensure (CUDA_SUCCESS == Init::result());
			ensure (CUDA_SUCCESS == cuDeviceGetCount(&count));
			ensure (ordinal < count);
			ensure (CUDA_SUCCESS == cuDeviceGet(&dev_, ordinal));
		}
		~Device()
		{
		}
		operator CUdevice()
		{
			return dev_;
		}
	};

	// context from cuCtxCreate - CUDA process
	class Ctx {
		CUcontext ctx_;
		Ctx(const Ctx&);
		Ctx& operator=(const Ctx&);
	public:
		Ctx(unsigned int flags = 0, CUdevice device = 0)
		{
			ensure (CUDA_SUCCESS == cuCtxCreate(&ctx_, flags, device));
		}
		~Ctx()
		{
			// destructor should not throw
			cuCtxDestroy(ctx_);
		}
		operator CUcontext()
		{
			return ctx_;
		}
	};

	// module with functions to launch on device
	class Module {
		CUmodule mod_;
		Module(const Module&);
		Module& operator=(Module&);
	public:
		// load from file
		struct Load 
		{
			const char* file_;
			Load(const char* file)
				: file_(file)
			{ }
		};
		// load from program data
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
			ensure (CUDA_SUCCESS == cuModuleLoad(&mod_, file.file_));
		}
		Module(const LoadData& data)
		{
			ensure (CUDA_SUCCESS == cuModuleLoadData(&mod_, data.data_));
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
			cuModuleUnload(mod_);
		}
		CUfunction GetFunction(const char* name)
		{
			CUfunction f;

			ensure (CUDA_SUCCESS == cuModuleGetFunction(&f, mod_, name));

			return f;
		}
	};

	// asynchronous execution streams on device
	class Stream {
		CUstream str_;
		Stream(const Stream&);
		Stream& operator=(const Stream&);
	public:
		Stream(unsigned int flags = 0)
		{
			ensure (CUDA_SUCCESS == cuStreamCreate(&str_, flags));
		}
		~Stream()
		{
			cuStreamDestroy(str_);
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
			ensure (CUDA_SUCCESS == cuEventCreate(&event_, flags));
		}
		~Event()
		{
			cuEventDestroy(event_);
		}
		operator CUevent()
		{
			return event_;
		}
	};

	inline unsigned int align(unsigned int o, unsigned int a) 
	{
		return (o + a - 1)&~(a - 1); // ALIGN_UP(o, a); ??? 
	}
	
	// execute function on device
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
		// this destructor throws so be sure to catch
		~Launch()
		{
			// launch when it goes out of scope
			ensure (CUDA_SUCCESS == cuParamSetSize(func_, off_));

			if (width_ == 0 || height_ == 0) {
				ensure (CUDA_SUCCESS == cuLaunch(func_));
			}
			else if (str_ == 0) {
				ensure (CUDA_SUCCESS == cuLaunchGrid(func_, width_, height_));
			}
			else {
				ensure (CUDA_SUCCESS == cuLaunchGridAsync(func_, width_, height_, str_));
			}
		}
		Launch& SetBlockShape(int x, int y = 1, int z = 1)
		{
			ensure (CUDA_SUCCESS == cuFuncSetBlockShape(func_, x, y, z));

			return *this;	
		}
		Launch& SetSharedSize(unsigned int bytes)
		{
			ensure (CUDA_SUCCESS == cuFuncSetSharedSize(func_, bytes));

			return *this;	
		}
		Launch& ParamSet(void* ptr)
		{
			off_ = align(off_, __alignof(ptr));
			ensure (CUDA_SUCCESS == cuParamSetv(func_, off_, ptr, sizeof(ptr)));
			off_ += sizeof(ptr);

			return *this;
		}
		Launch& ParamSet(unsigned int i)
		{
			off_ = align(off_, __alignof(i));
			ensure (CUDA_SUCCESS == cuParamSeti(func_, off_, i));
			off_ += sizeof(unsigned int);

			return *this;
		}
		Launch& ParamSet(float f)
		{
			off_ = align(off_, __alignof(f));
			ensure (CUDA_SUCCESS == cuParamSetf(func_, off_, f));
			off_ += sizeof(float);

			return *this;
		}
	};

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
				ensure (CUDA_SUCCESS == cuMemHostAlloc((void**)&mem_, size_*sizeof(T), flags));
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
				ensure (CUDA_SUCCESS == cuMemAlloc(&ptr_, size*sizeof(T)));
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

				ensure (CUDA_SUCCESS == cuArrayCreate(&ptr_, &cad_));
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

				ensure (CUDA_SUCCESS == cuArray3DCreate(&ptr_, &cad_));
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
			cuTexRefDestroy(tex_);
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
		ensure (CUDA_SUCCESS == cuMemcpyHtoD(d, h, h.Size()));

		return h;
	}
	template<class T>
	const Mem::Host<T>& operator>>(const Mem::Host<T>& h, Mem::Array<T>& a)
	{
		ensure (h.Size() <= a.Size());
		ensure (CUDA_SUCCESS == cuMemcpyHtoA(a, 0, h, h.Size()));

		return h;
	}
	// etc...
	template<class T>
	Mem::Host<T>& operator<<(Mem::Host<T>& h, /*const*/ Mem::Device<T>& d)
	{
		ensure (h.Size() >= d.Size());
		ensure (CUDA_SUCCESS == cuMemcpyDtoH(h, d, d.Size())); // const_cast<CUdeviceptr>(d)

		return h;
	}
	template<class T>
	Mem::Host<T>& operator<<(Mem::Host<T>& h, const Mem::Array<T>& a)
	{
		ensure (h.Size() >= a.Size());
		ensure (CUDA_SUCCESS == cuMemcpyAtoH(h, a, 0, a.Size()));

		return h;
	}
	// etc...
} // namespace cu
