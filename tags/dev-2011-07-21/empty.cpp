// empty.cpp - sample program using driver api
// Copyright (c) 2011 KALX, LLC. All rights reserved. No warranty is made.
#include "cuda/cu.h"

using namespace cu;

#define WARPS_PER_CTA   8
#define THREADS_PER_CTA (32 * WARPS_PER_CTA)
#define WARPS_PER_SM    48
#define CTAS_PER_SM     (WARPS_PER_SM / WARPS_PER_CTA)
#define SMS_PER_GPU     15
#define CTAS_PER_GPU    (SMS_PER_GPU * CTAS_PER_SM)
#define RUNS            4
#define CTAS            (RUNS * CTAS_PER_GPU)


void
test0(void)
{
	using cu::Module;
	
	cu::Device dev;
	cu::Ctx ctx(CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, dev);
	cu::Event ev[2];
	cu::Module mod(cu::Module::Load("empty.cubin"));
//	Module mod(Module::Load("C:\\Users\\kal\\Code\\nxll\\8\\Debug\\empty.cubin"));

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
}

int
main(int ac, char* av[])
{
	test0();

	return 0;
}
