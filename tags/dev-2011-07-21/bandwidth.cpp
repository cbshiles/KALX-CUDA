// bandwidth.cpp - test CUDA bandwidth
// Copyright (c) 2011 KALX, LLC. All rights reserved. No warranty is made.
//#define EXCEL12
#include "xll/xll.h"
#include "cuda/cu.h"

#ifndef CATEGORY
#define CATEGORY _T("CUDA")
#endif

using namespace xll;
using namespace cu;

static int count = cu::Device().Count();

static AddInX xai_d2h_bandwidth(
	FunctionX(XLL_DOUBLEX, "?xll_d2h_bandwidth", _T("TEST.D2H.BANDWIDTH"))
	.Arg(XLL_LONGX, _T("Bytes"), _T("is the number bytes to transfer. "))
	.Arg(XLL_WORDX, _T("Iterations"), _T("is the number of times to perform the transfer."))
	.Category(CATEGORY)
	.FunctionHelp(_T("Return the bandwith in MB/s of device to host data copy."))
	.Documentation()
);
double WINAPI
xll_d2h_bandwidth(ULONG size, WORD iter)
{
#pragma XLLEXPORT
	float bw;
	cu::Ctx ctx;

	cu::Event start, stop;
	cu::Mem::Host<unsigned char> h(size);
	cu::Mem::Device<unsigned char> d(size);

	cuStreamQuery(0);     
	cuEventRecord(start, 0); 
	if (!iter) iter = 10;
	for (WORD i = 0; i < iter; ++i)
		h << d;
	cuEventRecord(stop, 0);     
	cuEventSynchronize(stop);   

	cuEventElapsedTime(&bw, start, stop);
	bw = (1e3f * size * iter)/(bw * (1<<20));

	return bw;
}

static AddInX xai_h2d_bandwidth(
	FunctionX(XLL_DOUBLEX, "?xll_h2d_bandwidth", _T("TEST.H2D.BANDWIDTH"))
	.Arg(XLL_LONGX, _T("Bytes"), _T("is the number bytes to transfer. "))
	.Arg(XLL_WORDX, _T("Iterations"), _T("is the number of times to perform the transfer."))
	.Category(CATEGORY)
	.FunctionHelp(_T("Return the bandwith in MB/s of host to device data copy."))
	.Documentation()
);
double WINAPI
xll_h2d_bandwidth(ULONG size, WORD iter)
{
#pragma XLLEXPORT
	float bw;
	cu::Ctx ctx;

	cu::Event start, stop;
	cu::Mem::Host<unsigned char> h(size);
	cu::Mem::Device<unsigned char> d(size);

	cuStreamQuery(0);     
	cuEventRecord(start, 0); 
	if (!iter) iter = 10;
	for (WORD i = 0; i < iter; ++i)
		h >> d;
	cuEventRecord(stop, 0);     
	cuEventSynchronize(stop);   

	cuEventElapsedTime(&bw, start, stop);
	bw = (1e3f * size * iter)/(bw * (1<<20));

	return bw;
}