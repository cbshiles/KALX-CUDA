// marsaglia.h - George Marsaglia's random number generators
// Copyright (c) 2011 KALX, LLC. All rights reserved. No warranty is made.
// http://csis.hku.hk/~diehard 

#pragma once

namespace marsaglia {

	inline
	unsigned long cong(unsigned long seed = 0) 
	{ 
		static unsigned long x = 123456789;

		if (seed)
			x = seed;

		return x = 69069*x + 362437;
	} 

	inline
	unsigned long xorshift(unsigned long* seed = 0) 
	{
		static unsigned long x=123456789,y=362436069,z=521288629,w=88675123,v=886756453; 

		if (seed) {
			x = seed[0];
			y = seed[1];
			z = seed[2];
			w = seed[3];
			v = seed[4];
		}

		unsigned long t; 
		t=(x^(x>>7)); x=y; y=z; z=w; w=v; 
		v=(v^(v<<6))^(t^(t<<13)); 
	
		return (y+y+1)*v;
	} 

	inline
	unsigned long MWC256(unsigned long* seed = 0) // seed[256]
	{
		static unsigned long Q[256] = {0}, c=362436;

		if (seed) {
			for (int i = 0; i < 256; ++i)
				Q[i] = seed[i];
		}
		else if (!Q[0]) {
			for (int i = 0; i < 256; ++i)
				Q[i] = cong();
		}

		unsigned long long t,a=809430660LL; 
		static unsigned char i=255; 
		t=a*Q[++i]+c; c=(t>>32); 

		return Q[i]=t;      
	} 

	/*
	Here is a complimentary-multiply-with-carry RNG 
	with k=4097 and a near-record period, more than 
	10^33000 times as long as that of the Twister. 
	(2^131104 vs. 2^19937) 
	*/
	inline
	unsigned long CMWC4096(unsigned long* seed)
	{ 
		static unsigned long Q[4096] = {0},c=362436;
		unsigned long long t, a=18782LL; 
		static unsigned long i=4095; 
		unsigned long x,r=0xfffffffe; 

		if (seed) {
			for (int i = 0; i < 4096; ++i)
				Q[i] = seed[i];
		}
		else if (!Q[0]) {
			for (int i = 0; i < 4096; ++i)
				Q[i] = cong();
		}

		i=(i+1)&4095; 
		t=a*Q[i]+c; 
		c=(t>>32); x=t+c; if(x<c){x++;c++;} 

		return Q[i]=r-x;    
	} 

}



