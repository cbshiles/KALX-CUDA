// empty.cpp - sample program using driver api
// Copyright (c) 2011 KALX, LLC. All rights reserved. No warranty is made.
// Based on http://timothylottes.blogspot.com/2010/05/cuda-driver-api-example.html
#include <iostream>
#include "cu.h"

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
test(void)
{
	Device dev; // calls cuInit
	Ctx ctx(CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, dev);
	Event ev[2];
	Module mod(Module::Load("empty.cubin"));

	CUfunction f = mod.GetFunction("Empty");
//	cuFuncSetBlockShape(f, THREADS_PER_CTA, 1, 1);    
//	cuFuncSetSharedSize(f, 0);

	Mem::Device<int> clkD(CTAS);

	cuStreamQuery(0);     
	cuEventRecord(ev[0], 0);    

	Launch(f, CTAS, 1) // temporary object
		.SetBlockShape(THREADS_PER_CTA, 1, 1)
		.SetSharedSize(0)
		.ParamSet(&clkD);

	cuEventRecord(ev[1], 0);     
	cuEventSynchronize(ev[1]);     

	float ms;     
	cuEventElapsedTime(&ms, ev[0], ev[1]);

	Mem::Host<int> clkH(CTAS);
	clkH << clkD;

	unsigned long long int avgClk = 0;
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
	try {
		test();
	}
	catch (const std::exception& ex) {
		std::cerr << ex.what();

		return -1;
	}

	return 0;
}
