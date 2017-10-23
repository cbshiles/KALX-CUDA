// cu.cpp - CUDA driver API wrappers
// Copyright (c) 2011 KALX, LLC. All rights reserved. No warranty is made.
//#define EXCEL12
#include "xll/xll.h"
#include "cuda/ieeeconvert.h"
#include "cuda/cu.h"

#ifndef CATEGORY
#define CATEGORY _T("CUDA")
#endif

using namespace xll;

typedef traits<XLOPERX>::xfp xfp;
typedef traits<XLOPERX>::xword xword;

#define CUDA_HANDLE(h) h
#define HANDLE_CUDA(h) (int)h

static cu::Init anything;
static CUresult result;

// Device Management API
static AddInX xai_cuDeviceGetCount(
	FunctionX(XLL_LONGX, _T("?xll_cuDeviceGetCount"), _T("cuDeviceGetCount"))
	.Category(CATEGORY)
	.FunctionHelp(_T("Returns the number of compute-capable devices."))
	.Documentation()
);
LONG WINAPI
xll_cuDeviceGetCount(void)
{
#pragma XLLEXPORT
	int count(-1);
	CUresult result;

	result = cuDeviceGetCount(&count);
	if (CUDA_SUCCESS != result)
		XLL_ERROR(_T("cuDeviceGetCount failed")); 

	return count;
}

static AddInX xai_cuDriverGetVersion(
	FunctionX(XLL_LONGX, _T("?xll_cuDriverGetVersion"), _T("cuDriverGetVersion"))
	.Category(CATEGORY)
	.FunctionHelp(_T("Returns the CUDA driver version."))
	.Documentation()
);
LONG WINAPI
xll_cuDriverGetVersion(void)
{
#pragma XLLEXPORT
	int count(-1);
	CUresult result;

	result = cuDriverGetVersion(&count);
	if (CUDA_SUCCESS != result)
		XLL_ERROR(_T("cuDriverGetVersion failed")); 

	return count;
}

static AddInX xai_cuDeviceGet(
	FunctionX(XLL_HANDLEX, _T("?xll_cuDeviceGet"), _T("cuDeviceGet"))
	.Arg(XLL_SHORTX, _T("Ordinal"), _T("Device number to get handle for. "))
	.Category(CATEGORY)
	.FunctionHelp(_T("Returns the device handle. "))
	.Documentation()
);
HANDLEX WINAPI
xll_cuDeviceGet(SHORT ordinal)
{
#pragma XLLEXPORT
	CUresult result;
	CUdevice device;
	
	result = cuDeviceGet(&device, ordinal);
	if (CUDA_SUCCESS != result)
		XLL_ERROR(_T("cuDeviceGet failed"));
	
	return CUDA_HANDLE(device);
}

static AddInX xai_cuDeviceComputeCapability(
	FunctionX(XLL_DOUBLEX, _T("?xll_cuDeviceComputeCapability"), _T("cuDeviceComputeCapability"))
	.Arg(XLL_HANDLEX, _T("Handle"), _T("Handle to device returned by cuDeviceGet. "))
	.Category(CATEGORY)
	.FunctionHelp(_T("Returns the major.minor capability of the device. "))
	.Documentation()
);
double WINAPI
xll_cuDeviceComputeCapability(HANDLEX device)
{
#pragma XLLEXPORT
	CUresult result;
	int major(0), minor(0);
	
	result = cuDeviceComputeCapability(&major, &minor, HANDLE_CUDA(device));
	if (CUDA_SUCCESS != result)
		XLL_ERROR(_T("cuDeviceComputeCapability failed"));
	
	return major + minor/10.;
}

static AddInX xai_cuDeviceGetName(
	FunctionX(XLL_CSTRINGX, _T("?xll_cuDeviceGetName"), _T("cuDeviceGetName"))
	.Arg(XLL_HANDLEX, _T("Handle"), _T("Handle to device returned by cuDeviceGet. "))
	.Category(CATEGORY)
	.FunctionHelp(_T("Returns an identifer string for the device. "))
	.Documentation()
);
LPCTSTR WINAPI
xll_cuDeviceGetName(HANDLEX device)
{
#pragma XLLEXPORT
	CUresult result;
	static TCHAR name[256];
	
	name[0] = 0;
	result = cuDeviceGetName(name, 256, HANDLE_CUDA(device));
	if (CUDA_SUCCESS != result)
		XLL_ERROR(_T("cuDeviceGetName failed"));
	
	return name;
}

static AddInX xai_cuDeviceTotalMem(
	FunctionX(XLL_DOUBLEX, _T("?xll_cuDeviceTotalMem"), _T("cuDeviceTotalMem"))
	.Arg(XLL_HANDLEX, _T("Handle"), _T("Handle to device returned by cuDeviceGet. "))
	.Category(CATEGORY)
	.FunctionHelp(_T("Returns the total amount of memory in bytes on the device. "))
	.Documentation()
);
double WINAPI
xll_cuDeviceTotalMem(HANDLEX device)
{
#pragma XLLEXPORT
	CUresult result;
	size_t size(0);
	
	result = cuDeviceTotalMem(&size, HANDLE_CUDA(device));
	if (CUDA_SUCCESS != result)
		XLL_ERROR(_T("cuDeviceTotalMem failed"));
	
	return size;
}

XLL_ENUM(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, CATEGORY, _T("Maximum number of threads per block."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, CATEGORY, _T("Maximum block dimension X."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, CATEGORY, _T("Maximum block dimension Y."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, CATEGORY, _T("Maximum block dimension Z."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, CATEGORY, _T("Maximum grid dimension X."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, CATEGORY, _T("Maximum grid dimension Y."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, CATEGORY, _T("Maximum grid dimension Z."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, CATEGORY, _T("Maximum shared memory available per block in bytes."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, CATEGORY, _T("Deprecated."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, CATEGORY, _T("Memory available on device for __constant__ variables in a CUDA C kernel in bytes."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_WARP_SIZE, CU_DEVICE_ATTRIBUTE_WARP_SIZE, CATEGORY, _T("Warp size in threads."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_MAX_PITCH, CU_DEVICE_ATTRIBUTE_MAX_PITCH, CATEGORY, _T("Maximum pitch in bytes allowed by memory copies."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, CATEGORY, _T("Maximum number of 32-bit registers available per block."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK, CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK, CATEGORY, _T("Deprecated."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_CLOCK_RATE, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, CATEGORY, _T("Peak clock frequency in kilohertz."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, CATEGORY, _T("Alignment requirement for textures."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, CATEGORY, _T("Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT.."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, CATEGORY, _T("Number of multiprocessors on device."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, CATEGORY, _T("Specifies whether there is a run time limit on kernels."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_INTEGRATED, CU_DEVICE_ATTRIBUTE_INTEGRATED, CATEGORY, _T("Device is integrated with host memory."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, CATEGORY, _T("Device can map host memory into CUDA address space."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, CATEGORY, _T("Compute mode (See ::CUcomputemode for details)."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, CATEGORY, _T("Maximum 1D texture width."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, CATEGORY, _T("Maximum 2D texture width."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, CATEGORY, _T("Maximum 2D texture height."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, CATEGORY, _T("Maximum 3D texture width."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, CATEGORY, _T("Maximum 3D texture height."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, CATEGORY, _T("Maximum 3D texture depth."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH, CATEGORY, _T("Deprecated."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT, CATEGORY, _T("Deprecated."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES, CATEGORY, _T("Deprecated."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, CATEGORY, _T("Alignment requirement for surfaces."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, CATEGORY, _T("Device can possibly execute multiple kernels concurrently."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_ECC_ENABLED, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, CATEGORY, _T("Device has ECC support enabled."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, CATEGORY, _T("PCI bus ID of the device."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, CATEGORY, _T("PCI device ID of the device."));
XLL_ENUM(CU_DEVICE_ATTRIBUTE_TCC_DRIVER, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, CATEGORY, _T("Device is using TCC driver model."));

static AddInX xai_cuDeviceGetAttribute(
	FunctionX(XLL_LONGX, _T("?xll_cuDeviceGetAttribute"), _T("cuDeviceGetAttribute"))
	.Arg(XLL_SHORTX, _T("Attribute"), _T("Device attribute to query from the CU_DEVICE_ATTRIBUTE_* enumeration."))
	.Arg(XLL_HANDLEX, _T("Handle"), _T("Handle to device returned by cuDeviceGet. "))
	.Category(CATEGORY)
	.FunctionHelp(_T("Returns information about the device. "))
	.Documentation()
);
LONG WINAPI
xll_cuDeviceGetAttribute(CUdevice_attribute_enum attr, HANDLEX device)
{
#pragma XLLEXPORT
	CUresult result;
	int value(0);
	
	result = cuDeviceGetAttribute(&value, attr, HANDLE_CUDA(device));
	if (CUDA_SUCCESS != result)
		XLL_ERROR(_T("cuDeviceGetAttribute failed"));
	
	return value;
}

// Context flags
XLL_ENUM(CU_CTX_SCHED_AUTO, CU_CTX_SCHED_AUTO, CATEGORY, 
	_T("Automatic scheduling. "));
XLL_ENUM(CU_CTX_SCHED_SPIN, CU_CTX_SCHED_SPIN, CATEGORY, 
	_T("Set spin as default scheduling. "));
XLL_ENUM(CU_CTX_SCHED_YIELD, CU_CTX_SCHED_YIELD, CATEGORY, 
	_T("Set yield as default scheduling. "));
XLL_ENUM(CU_CTX_BLOCKING_SYNC, CU_CTX_BLOCKING_SYNC, CATEGORY,
	_T("Set blocking synchronization as default scheduling (deprecated)."));
XLL_ENUM(CU_CTX_MAP_HOST, CU_CTX_MAP_HOST, CATEGORY,
	_T("Support mapped pinned allocations."));
XLL_ENUM(CU_CTX_LMEM_RESIZE_TO_MAX, CU_CTX_LMEM_RESIZE_TO_MAX, CATEGORY,
	_T("Keep local memory allocation after launch."));

static AddInX xai_cuCtx(
	FunctionX(XLL_HANDLEX, _T("?xll_cuCtx"), _T("cuCtx"))
	.Arg(XLL_WORDX, _T("Flags"), _T("Create an event using flags from the CU_CTX_SCHED_* enumeration")) 
	.Arg(XLL_HANDLEX XLL_UNCALCED, _T("Handle"), _T("Handle to device returned by cuDeviceGet "))
	.Category(CATEGORY)
	.FunctionHelp(_T("Creates a new CUDA context and associates it with the calling thread."))
	.Documentation(
		_T("The <codeInline>CU_CTX_SCHED_*</codeInline> flags are bit masks that ")
		_T("can be added. They come before the handle in the argument list ")
		_T("just like the call to <codeInline>cuCtxCreate</codeInline>.")
	)
);
HANDLEX WINAPI
xll_cuCtx(WORD flags, HANDLEX device)
{
#pragma XLLEXPORT
	handle<cu::Ctx> h;
	
	try {
		h = new cu::Ctx(flags, HANDLE_CUDA(device));
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return 0;
	}

	return h.get();
}

static AddInX xai_cuStream(
	FunctionX(XLL_HANDLEX XLL_UNCALCED, _T("?xll_cuStream"), _T("cuStream"))
	.Category(CATEGORY)
	.FunctionHelp(_T("Creates a stream."))
	.Documentation()
);
HANDLEX WINAPI
xll_cuStream()
{
#pragma XLLEXPORT
	handle<cu::Stream> h;
	
	try {
		h = new cu::Stream();
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return 0;
	}

	return h.get();
}

static AddInX xai_cuStreamQuery(
	FunctionX(XLL_DOUBLEX, _T("?xll_cuStreamQuery"), _T("cuStreamQuery"))
	.Arg(XLL_HANDLEX, _T("stream"), _T("is a handle to the starting event. ")) 
	.Category(CATEGORY)
	.FunctionHelp(_T("Query stream."))
	.Documentation()
);
HANDLEX WINAPI
xll_cuStreamQuery(HANDLEX stream)
{
#pragma XLLEXPORT
	try {
		handle<cu::Stream> stream_(stream);
		ensure (stream_);
		ensure (CUDA_SUCCESS == (result = cuStreamQuery(*stream_)));
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return 0;
	}

	return stream;
}

static AddInX xai_cuStreamSynchronize(
	FunctionX(XLL_DOUBLEX, _T("?xll_cuStreamSynchronize"), _T("cuStreamSynchronize"))
	.Arg(XLL_HANDLEX, _T("stream"), _T("is a handle to the starting event. ")) 
	.Category(CATEGORY)
	.FunctionHelp(_T("Synchronize stream ."))
	.Documentation()
);
HANDLEX WINAPI
xll_cuStreamSynchronize(HANDLEX stream)
{
#pragma XLLEXPORT
	try {
		ensure (CUDA_SUCCESS == (result = cuStreamSynchronize(*handle<cu::Stream>(stream))));
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return 0;
	}

	return stream;
}

XLL_ENUM(CU_EVENT_DEFAULT, CU_EVENT_DEFAULT, CATEGORY, 
	_T("Default event creation flag."));
XLL_ENUM(CU_EVENT_BLOCKING_SYNC, CU_EVENT_BLOCKING_SYNC, CATEGORY, 
	_T("Specifies that event should use blocking synchronization. "));

static AddInX xai_cuEvent(
	FunctionX(XLL_HANDLEX XLL_UNCALCED, _T("?xll_cuEvent"), _T("cuEvent"))
	.Arg(XLL_WORDX, _T("Flags"), _T("are flags from the CU_EVENT_* enumeration. ")) 
	.Category(CATEGORY)
	.FunctionHelp(_T("Creates an event with specified Flags."))
	.Documentation(
		_T("A CPU thread that uses <codeInline>cuEventSynchronize</codeInline> ")
		_T("to wait on an event created with this flag will block until the event ")
		_T("has actually been recorded.")
	)
);
HANDLEX WINAPI
xll_cuEvent(WORD flags)
{
#pragma XLLEXPORT
	handle<cu::Event> h;
	
	try {
		h = new cu::Event(flags);
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}

	return h.get();
}

static AddInX xai_cuEventElapsedTime(
	FunctionX(XLL_DOUBLEX, _T("?xll_cuEventElapsedTime"), _T("cuEventElapsedTime"))
	.Arg(XLL_HANDLEX, _T("Start"), _T("is a handle to the starting event. ")) 
	.Arg(XLL_HANDLEX, _T("Stop"), _T("is a handle to the stopping event. ")) 
	.Category(CATEGORY)
	.FunctionHelp(_T("Computes the elapsed time between two events in milliseconds ."))
	.Documentation()
);
double WINAPI
xll_cuEventElapsedTime(HANDLEX start, HANDLEX stop)
{
#pragma XLLEXPORT
	float ms(std::numeric_limits<float>::quiet_NaN());

	try {
		handle<cu::Event> start_(start);
		ensure (start_);
		handle<cu::Event> stop_(stop);
		ensure (stop_);
		ensure (CUDA_SUCCESS == (result = cuEventElapsedTime(&ms, *start_, *stop_)));
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return std::numeric_limits<float>::quiet_NaN();
	}

	return ms;
}

static AddInX xai_cuEventRecord(
	FunctionX(XLL_HANDLEX, _T("?xll_cuEventRecord"), _T("cuEventRecord"))
	.Arg(XLL_HANDLEX, _T("Event"), _T("is a handle to an event. ")) 
	.Arg(XLL_HANDLEX, _T("Stream"), _T("is a handle to a stream. ")) 
	.Category(CATEGORY)
	.FunctionHelp(_T("Records an event. "))
	.Documentation(
		_T("Records an event. If stream is non-zero, the event is recorded ")
		_T("after all preceding operations in the stream have been completed; ")
		_T("otherwise, it is recorded after all preceding operations in the CUDA ")
		_T("context have been completed. Since operation is asynchronous, ")
		_T("<codeInline>cuEventQuery</codeInline> and/or ")
		_T("<codeInline>cuEventSynchronize</codeInline> must be used to determine ")
		_T("when the event has actually been recorded.")
	)
);
HANDLEX WINAPI
xll_cuEventRecord(HANDLEX event, HANDLEX stream)
{
#pragma XLLEXPORT
	try {
		handle<cu::Event> event_(event);
		ensure (event_);
		handle<cu::Stream> stream_(stream);
		ensure (stream_);
		ensure (CUDA_SUCCESS == (result = cuEventRecord(*event_, *stream_)));
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		event = 0;
	}

	return event;
}

static AddInX xai_cuEventSynchronize(
	FunctionX(XLL_DOUBLEX, _T("?xll_cuEventSynchronize"), _T("cuEventSynchronize"))
	.Arg(XLL_HANDLEX, _T("Event"), _T("is a handle to an event. ")) 
	.Category(CATEGORY)
	.FunctionHelp(_T("Waits for an event to complete. "))
	.Documentation(
		_T("Waits until the event has actually been recorded. If ")
		_T("<codeInline>cuEventRecord</codeInline> has been called on this event, the ")
		_T("function returns <codeInline>CUDA_ERROR_INVALID_VALUE</codeInline>. Waiting ")
		_T("for an event that was created with the <codeInline>CU_EVENT_BLOCKING_SYNC</codeInline> ")
		_T("flag will cause the calling CPU thread to block until the event has actually been recorded.")
	)
);
HANDLEX WINAPI
xll_cuEventSynchronize(HANDLEX event)
{
#pragma XLLEXPORT
	

	try {
		handle<cu::Event> event_(event);
		ensure (event_);
		ensure (CUDA_SUCCESS == (result = cuEventSynchronize(*event_)));
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return 0;
	}

	return event;
}

static AddIn xai_cuModuleLoad(
	Function(XLL_HANDLEX, "?xll_cuModuleLoad", "cuModuleLoad")
	.Arg(XLL_CSTRING, _T("Name"), _T("is the name of the file containg the module."))
	.Category(CATEGORY)
	.FunctionHelp(_T("Open file with the given Name and loads the corresponding module module into the current context."))
	.Documentation(
		_T("The CUDA driver API does not attempt to lazily allocate the resources ")
		_T("needed by a module; if the memory for functions and data (constant and global) ")
		_T("needed by the module cannot be allocated, <codeInline>cuModuleLoad</codeInline> fails. ")
		_T("The file should be a cubin file as output by nvcc or a PTX file, either as output by nvcc ")
		_T("or handwritten.")
	)
);
HANDLEX WINAPI
xll_cuModuleLoad(const char* name)
{
#pragma XLLEXPORT
	handle<cu::Module> h;
	
	try{
		ensure (0 != (h = new cu::Module(cu::Module::Load(name))));
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return 0;
	}

	return h.get();
}

static AddIn xai_cuModuleGetFunction(
	Function(XLL_HANDLEX, "?xll_cuModuleGetFunction", "cuModuleGetFunction")
	.Arg(XLL_HANDLEX, _T("Module"), _T("is a handle to a module."))
	.Arg(XLL_CSTRING, _T("Name"), _T("is the name of the funtion in the module."))
	.Category(CATEGORY)
	.FunctionHelp(_T("Returns the handle of the function Name located in Module."))
	.Documentation(
		_T("Returns 0 if no such function exists in the module.")
	)
);
HANDLEX WINAPI
xll_cuModuleGetFunction(HANDLEX module, const char* name)
{
#pragma XLLEXPORT
	CUfunction f(0);
	
	try{
		ensure (0 != (f = handle<cu::Module>(module)->GetFunction(name)));
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return 0;
	}

	return p2h<CUfunc_st>(f);
}

// unsigned int and float encoding
static AddInX xai_cuFloat(
	FunctionX(XLL_DOUBLEX, _T("?xll_cuFloat"), _T("cuSetFloat"))
	.Arg(XLL_DOUBLE, _T("Num"), _T("is a number to be converted to a device floating point value."))
	.Category(CATEGORY)
	.FunctionHelp(_T("Convert host double to device float"))
	.Documentation(_T("Same as <codeInline>cudaSetDoubleForDevice</codeInline>."))
);
double WINAPI
xll_cuFloat(double d)
{
#pragma XLLEXPORT
	convert2float(d);

	return d;
}
static AddInX xai_cuUInt(
	FunctionX(XLL_DOUBLEX, _T("?xll_cuUInt"), _T("cuSetUInt"))
	.Arg(XLL_DOUBLE, _T("Num"), _T("is a number int to be converted to a device unsigned int value."))
	.Category(CATEGORY)
	.FunctionHelp(_T("Convert host unsigned int to device unsigned int"))
	.Documentation(_T("Used as an argument <codeInline>to cuLaunch</codeInline>."))
);
double WINAPI
xll_cuUInt(double d)
{
#pragma XLLEXPORT
	union { double d; unsigned int u[2]; } du;

	du.u[0] = 0;
	du.u[1] = static_cast<unsigned int>(d);

	return du.d;
}

static AddInX xai_cuLaunch(
	FunctionX(XLL_HANDLEX, _T("?xll_cuLaunch"), _T("cuLaunch"))
	.Arg(XLL_HANDLE, _T("Function"), _T("is a handle the a function returned by cuModuleGetFunction."))
	.Arg(XLL_FPX, _T("Args"), _T("is the array of arguments the function takes."))
	.Arg(XLL_FPX, _T("Grid"), _T("is the grid dimensions to launch."))
	.Arg(XLL_FPX, _T("Block"), _T("is the block shape to use."))
	.Arg(XLL_LONGX, _T("Shared"), _T("is the number of bytes of dynamic shared memory that will be available to each thread block."))
	.Category(CATEGORY)
	.FunctionHelp(_T("Invoke a kernel on the device."))
	.Documentation()
);
HANDLEX WINAPI
xll_cuLaunch(HANDLEX f, const xfp* pa, const xfp* pg, const xfp* pb, LONG s)
{
#pragma XLLEXPORT
	HANDLEX h(0);

	try {
		int w, h(0);

		w = static_cast<int>(pg->array[0]);
		if (size(*pg) > 1)
			h = static_cast<int>(pg->array[1]);

		cu::Launch k(h2p<CUfunc_st>(f), w, h);

		if (pa->array[0] != 0) {
			union { double d; unsigned int u[2]; } du;
			for (xword i = 0; i < size(*pa); ++i) {
				du.d = pa->array[i];
				if (du.u[0] == 0) { // uint
					k.ParamSet(du.u[1]);
				}
				else if (du.u[1] == 0) { // float
					du.d = pa->array[i];
					k.ParamSet(convert2double(du.d));
				}
				else { // handle
					k.ParamSet(h2p<void>(pa->array[i]));
				}
			}
		}

		if (pb->array[0] != 0) {
			int x, y(1), z(1);

			x = static_cast<int>(pb->array[0]);
			if (size(*pb) > 1) {
				y = static_cast<int>(pb->array[1]);
			}
			if (size(*pb) > 2) {
				z = static_cast<int>(pb->array[2]);
			}

			k.SetBlockShape(x, y, z);
		}

		if (s) {
			k.SetSharedSize(s);
		}
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		h = 0;
	}

	// synchronize???

	return h;
}

// memory
static AddInX xai_mem_host_int(
	FunctionX(XLL_HANDLEX XLL_UNCALCEDX, _T("?xll_cuMemHostInt"), _T("cuMemHostInt"))
	.Arg(XLL_WORDX, _T("size"), _T("is the number of ints to allocate on the host"))
	.Category(CATEGORY)
	.FunctionHelp(_T("Return a handle to <codeInline>size</codeInline> 32-bit integers allocated on the host."))
	.Documentation()
);
HANDLEX WINAPI
xll_cuMemHostInt(WORD n)
{
#pragma XLLEXPORT
	handle<cu::Mem::Host<int>> h;

	try {
		ensure(h = new cu::Mem::Host<int>(n));
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return 0;
	}

	return h.get();
}
static AddInX xai_MemDeviceInt(
	FunctionX(XLL_HANDLEX XLL_UNCALCEDX, _T("?xll_cuMemDeviceInt"), _T("cuMemDeviceInt"))
	.Arg(XLL_WORDX, _T("size"), _T("is the number of ints to allocate on the device"))
	.Category(CATEGORY)
	.FunctionHelp(_T("Return a handle to <codeInline>size</codeInline> 32-bit integers allocated on the device."))
	.Documentation()
);
HANDLEX WINAPI
xll_cuMemDeviceInt(WORD n)
{
#pragma XLLEXPORT
	handle<cu::Mem::Device<int>> h;

	try {
		ensure (h = new cu::Mem::Device<int>(n));
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return 0;
	}

	return h.get();
}

static AddInX xai_cuMemcpyHtoD(
	FunctionX(XLL_BOOLX, _T("?xll_cuMemcpyHtoD"), _T("cuMemcpyHtoD"))
	.Arg(XLL_HANDLEX, _T("dev"), _T("is a handle to device memory"))
	.Arg(XLL_HANDLEX, _T("host"), _T("is a handle to host memory "))
	.Category(CATEGORY)
	.FunctionHelp(_T("Copy memory from host to device. "))
	.Documentation()
);
BOOL WINAPI
xll_cuMemcpyHtoD(HANDLEX dev, HANDLEX host)
{
#pragma XLLEXPORT
	try {
		// only int for now - add type info to Mem::XXX classes???
		handle<cu::Mem::Device<int>> dev_(dev);
		ensure (dev_);
		handle<cu::Mem::Host<int>> host_(host);
		ensure (host_);
		ensure (dev_->size() >= host_->size());
		ensure (CUDA_SUCCESS == (result = cuMemcpyHtoD(*dev_, *host_, host_->size())));
		
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return FALSE;
	}

	return TRUE;
}
static AddInX xai_cuMemcpyDtoH(
	FunctionX(XLL_BOOLX, _T("?xll_cuMemcpyDtoH"), _T("cuMemcpyDtoH"))
	.Arg(XLL_HANDLEX, _T("host"), _T("is a handle to host memory "))
	.Arg(XLL_HANDLEX, _T("dev"), _T("is a handle to device memory"))
	.Category(CATEGORY)
	.FunctionHelp(_T("Copy memory from device to host. "))
	.Documentation()
);
BOOL WINAPI
xll_cuMemcpyDtoH(HANDLEX host, HANDLEX dev)
{
#pragma XLLEXPORT
	try {
		// only int for now - add type info to Mem::XXX classes???
		handle<cu::Mem::Host<int>> host_(host);
		ensure (host_);
		handle<cu::Mem::Device<int>> dev_(dev);
		ensure (dev_);
		ensure (dev_->size() <= host_->size());
		CUresult result = cuMemcpyDtoH(*host_, *dev_, host_->size());
		ensure (result == CUDA_SUCCESS);
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return FALSE;
	}

	return TRUE;
}

#ifdef _DEBUG

using namespace cu;

#define WARPS_PER_CTA   8
#define THREADS_PER_CTA (32 * WARPS_PER_CTA)
#define WARPS_PER_SM    48
#define CTAS_PER_SM     (WARPS_PER_SM / WARPS_PER_CTA)
#define SMS_PER_GPU     15
#define CTAS_PER_GPU    (SMS_PER_GPU * CTAS_PER_SM)
#define RUNS            4
#define CTAS            (RUNS * CTAS_PER_GPU)

const char* VecAdd_ptx = 
".version 1.4\n"
"	.target sm_10, map_f64_to_f32\n"
"	// compiled with /usr/local/cuda/bin/../open64/lib//be\n"
"	// nvopencc 3.0 built on 2010-03-11\n"
"\n"
"	//-----------------------------------------------------------\n"
"	// Compiling /tmp/tmpxft_0000e98a_00000000-9_vectorAdd.compute_10.cpp3.i (/var/folders/fZ/fZpMfXGjEIefAb1MGdDCpk+++TI/-Tmp-/ccBI#.ZvYJOx)\n"
"	//-----------------------------------------------------------\n"
"\n"
"	//-----------------------------------------------------------\n"
"	// Options:\n"
"	//-----------------------------------------------------------\n"
"	//  Target:ptx, ISA:sm_10, Endian:little, Pointer Size:32\n"
"	//  -O3	(Optimization level)\n"
"	//  -g0	(Debug level)\n"
"	//  -m2	(Report advisories)\n"
"	//-----------------------------------------------------------\n"
"\n"
"\n"
"\n"
"	.entry _Z6VecAddPKfS0_Pfi (\n"
"		.param .u32 __cudaparm__Z6VecAddPKfS0_Pfi_A,\n"
"		.param .u32 __cudaparm__Z6VecAddPKfS0_Pfi_B,\n"
"		.param .u32 __cudaparm__Z6VecAddPKfS0_Pfi_C,\n"
"		.param .s32 __cudaparm__Z6VecAddPKfS0_Pfi_N)\n"
"	{\n"
"	.reg .u16 %rh<4>;\n"
"	.reg .u32 %r<13>;\n"
"	.reg .f32 %f<5>;\n"
"	.reg .pred %p<3>;\n"
"	.loc	28	43	0\n"
"$LBB1__Z6VecAddPKfS0_Pfi:\n"
"	mov.u16 	%rh1, %ctaid.x;\n"
"	mov.u16 	%rh2, %ntid.x;\n"
"	mul.wide.u16 	%r1, %rh1, %rh2;\n"
"	cvt.u32.u16 	%r2, %tid.x;\n"
"	add.u32 	%r3, %r2, %r1;\n"
"	ld.param.s32 	%r4, [__cudaparm__Z6VecAddPKfS0_Pfi_N];\n"
"	setp.le.s32 	%p1, %r4, %r3;\n"
"	@%p1 bra 	$Lt_0_1026;\n"
"	.loc	28	47	0\n"
"	mul.lo.u32 	%r5, %r3, 4;\n"
"	ld.param.u32 	%r6, [__cudaparm__Z6VecAddPKfS0_Pfi_A];\n"
"	add.u32 	%r7, %r6, %r5;\n"
"	ld.global.f32 	%f1, [%r7+0];\n"
"	ld.param.u32 	%r8, [__cudaparm__Z6VecAddPKfS0_Pfi_B];\n"
"	add.u32 	%r9, %r8, %r5;\n"
"	ld.global.f32 	%f2, [%r9+0];\n"
"	add.f32 	%f3, %f1, %f2;\n"
"	ld.param.u32 	%r10, [__cudaparm__Z6VecAddPKfS0_Pfi_C];\n"
"	add.u32 	%r11, %r10, %r5;\n"
"	st.global.f32 	[%r11+0], %f3;\n"
"$Lt_0_1026:\n"
"	.loc	28	48	0\n"
"	exit;\n"
"$LDWend__Z6VecAddPKfS0_Pfi:\n"
"	} // _Z6VecAddPKfS0_Pfi\n"
;

const char* ptx = 
".version 1.4"
".target sm_10, map_f64_to_f32"
"// compiled with C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v4.0\\\\bin/../open64/lib//be.exe"
"// nvopencc 4.0 built on 2011-03-24"
""
"//-----------------------------------------------------------"
"// Compiling empty.cpp3.i (C:/Users/kal/AppData/Local/Temp/ccBI#.a01636)"
"//-----------------------------------------------------------"
""
"//-----------------------------------------------------------"
"// Options:"
"//-----------------------------------------------------------"
"//  Target:ptx, ISA:sm_10, Endian:little, Pointer Size:32"
"//  -O3	(Optimization level)"
"//  -g0	(Debug level)"
"//  -m2	(Report advisories)"
"//-----------------------------------------------------------"
""
".file	1	\"empty.cudafe2.gpu\""
".file	2	\"C:\\Program Files\\Microsoft Visual Studio 8\\VC\\INCLUDE\\crtdefs.h\""
".file	3	\"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v4.0//include\\crt/device_runtime.h\""
".file	4	\"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v4.0//include\\host_defines.h\""
".file	5	\"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v4.0//include\\builtin_types.h\""
".file	6	\"c:\\program files\\nvidia gpu computing toolkit\\cuda\\v4.0\\include\\device_types.h\""
".file	7	\"c:\\program files\\nvidia gpu computing toolkit\\cuda\\v4.0\\include\\driver_types.h\""
".file	8	\"c:\\program files\\nvidia gpu computing toolkit\\cuda\\v4.0\\include\\surface_types.h\""
".file	9	\"c:\\program files\\nvidia gpu computing toolkit\\cuda\\v4.0\\include\\texture_types.h\""
".file	10	\"c:\\program files\\nvidia gpu computing toolkit\\cuda\\v4.0\\include\\vector_types.h\""
".file	11	\"c:\\program files\\nvidia gpu computing toolkit\\cuda\\v4.0\\include\\builtin_types.h\""
".file	12	\"c:\\program files\\nvidia gpu computing toolkit\\cuda\\v4.0\\include\\host_defines.h\""
".file	13	\"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v4.0//include\\device_launch_parameters.h\""
".file	14	\"c:\\program files\\nvidia gpu computing toolkit\\cuda\\v4.0\\include\\crt\\storage_class.h\""
".file	15	\"C:\\Program Files\\Microsoft Visual Studio 8\\VC\\INCLUDE\\time.h\""
".file	16	\"empty.cu\""
".file	17	\"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v4.0//include\\common_functions.h\""
".file	18	\"c:\\program files\\nvidia gpu computing toolkit\\cuda\\v4.0\\include\\math_functions.h\""
".file	19	\"c:\\program files\\nvidia gpu computing toolkit\\cuda\\v4.0\\include\\math_constants.h\""
".file	20	\"c:\\program files\\nvidia gpu computing toolkit\\cuda\\v4.0\\include\\device_functions.h\""
".file	21	\"c:\\program files\\nvidia gpu computing toolkit\\cuda\\v4.0\\include\\sm_11_atomic_functions.h\""
".file	22	\"c:\\program files\\nvidia gpu computing toolkit\\cuda\\v4.0\\include\\sm_12_atomic_functions.h\""
".file	23	\"c:\\program files\\nvidia gpu computing toolkit\\cuda\\v4.0\\include\\sm_13_double_functions.h\""
".file	24	\"c:\\program files\\nvidia gpu computing toolkit\\cuda\\v4.0\\include\\sm_20_atomic_functions.h\""
".file	25	\"c:\\program files\\nvidia gpu computing toolkit\\cuda\\v4.0\\include\\sm_20_intrinsics.h\""
".file	26	\"c:\\program files\\nvidia gpu computing toolkit\\cuda\\v4.0\\include\\surface_functions.h\""
".file	27	\"c:\\program files\\nvidia gpu computing toolkit\\cuda\\v4.0\\include\\texture_fetch_functions.h\""
".file	28	\"c:\\program files\\nvidia gpu computing toolkit\\cuda\\v4.0\\include\\math_functions_dbl_ptx1.h\""
""
""
".entry Empty ("
"	.param .u32 __cudaparm_Empty_clockOut)"
"{"
".reg .u16 %rh<3>;"
".reg .u32 %rv1;"
".reg .u32 %r<14>;"
".reg .pred %p<3>;"
".loc	16	3	0"
"$LDWbegin_Empty:"
".loc	16	5	0"
"mov.u32 	%r1, %clock;"
"mov.s32 	%r2, %r1;"
"mov.s32 	%r3, %r2;"
"cvt.u32.u16 	%r4, %tid.x;"
"mov.u32 	%r5, 0;"
"setp.ne.u32 	%p1, %r4, %r5;"
"@%p1 bra 	$Lt_0_1026;"
".loc	16	7	0"
"mov.u32 	%r6, %clock;"
"mov.s32 	%r7, %r6;"
"mov.s32 	%r8, %r3;"
"sub.u32 	%r9, %r7, %r8;"
"ld.param.u32 	%r10, [__cudaparm_Empty_clockOut];"
"mov.u16 	%rh1, %ctaid.x;"
"mul.wide.u16 	%r11, %rh1, 4;"
"add.u32 	%r12, %r10, %r11;"
"st.volatile.global.u32 	[%r12+0], %r9;"
"$Lt_0_1026:"
".loc	16	8	0"
"exit;"
"$LDWend_Empty:"
"} // Empty"
;

int
test0(void)
{
	using cu::Module;
	
	cu::Device dev;
	cu::Ctx ctx(CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, dev);
	cu::Event ev[2];
	cu::Module mod(Module::Load("empty.cubin"));
//	Module mod(Module::Load("C:\\Users\\kal\\Code\\nxll\\8\\Debug\\empty.cubin"));
//	Module mod(Module::LoadData(ptx));

	CUfunction f = mod.GetFunction("Empty");
	cuFuncSetBlockShape(f, THREADS_PER_CTA, 1, 1);    
	cuFuncSetSharedSize(f, 0);

	cu::Mem::Device<int> clkD(CTAS);
	cu::Mem::Host<int> clkH(CTAS);

	cuStreamQuery(0);     
	cuEventRecord(ev[0], 0);    
	cu::Launch(f, CTAS, 1).ParamSet(&clkD);
	cuEventRecord(ev[1], 0);     
	cuEventSynchronize(ev[1]);     
	float ms;     
	cuEventElapsedTime(&ms, ev[0], ev[1]);
	clkH << clkD;

	unsigned long long int avgClk = 0;
//	unsigned long long int ctaTotal = 0;
	unsigned int maxClk = 0;
	unsigned int minClk = 0xFFFFFFFF;
	unsigned int cta;
	for(cta = 0; cta < CTAS; cta++) {
		unsigned int clk = clkH[cta];
		if(clk > maxClk) maxClk = clk;
		if(clk < minClk) minClk = clk;
		avgClk += ((unsigned long long int)clk); 
	}
	avgClk = (unsigned int)(avgClk/((unsigned long long int)CTAS));

	return 1;
}
//Auto<OpenAfter> xao_test0(test0);


#endif // _DEBUG