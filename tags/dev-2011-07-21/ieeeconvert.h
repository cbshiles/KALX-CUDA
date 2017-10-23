// ieeeconvert.h - convert IEEE doubles to floats and vice versa
// Copyright (c) 2011 KALX, LLC. All rights reserved. No warranty is made.
// TODO: NaN, +-Inf, +-0, denormal
#include <cstdlib>
#include "xll/utility/ensure.h"

static_assert(sizeof(unsigned int) == 4, "ieeeconvert.h: unsigned ints must be 32 bits");
static_assert(sizeof(unsigned long long) == 8, "ieeeconvert.h: unsigned long long must be 64 bits");

#define IEEE_DSGN 0x8000000000000000ul
#define IEEE_DEXP 0x7FF0000000000000ul
#define IEEE_DMAN 0x000FFFFFFFFFFFFFul

#define IEEE_SSGN 0x80000000u
#define IEEE_SEXP 0x7F800000u
#define IEEE_SMAN 0x007FFFFFu

inline float double2float(double d)
{
	union { float f; unsigned int i; } fi;
	union { double d; unsigned long long l; } di;
	unsigned long long s, e, m; // sign, exponent, mantissa

	di.d = d;

	s = (di.l&IEEE_DSGN)>>63;
	e = (di.l&IEEE_DEXP)>>52;
	m = (di.l&IEEE_DMAN)>>29;

	fi.i = static_cast<unsigned int>((s<<31)|((e-1024+128)<<23)|m);

	return fi.f;
}

inline double float2double(float f)
{
	union { float f; unsigned int i; } fi;
	union { double d; unsigned long long l; } di;

	fi.f = f;
	unsigned long long s, e, b; // sign, exponent, fraction
	s = (fi.i&IEEE_SSGN)>>31;
	e = (fi.i&IEEE_SEXP)>>23;
	b = (fi.i&IEEE_SMAN);

	di.l = (s<<63)|((e-128+1024)<<52)|(b<<29);

	return di.d;
}

// stuff float in low bits of double
inline unsigned int convert2float(double& d)
{
	union { float f; unsigned int i; } fi;
	union { double d; unsigned int i[2]; } di;

	fi.f = double2float(d);
	di.i[0] = fi.i;
	di.i[1] = 0;

	d = di.d;

	return fi.i;
}

// extract float from low bits of double
inline float convert2double(double& d)
{
	union { float f; unsigned int i; } fi;
	union { double d; unsigned int i[2]; } di;

	di.d = d;
	fi.i = di.i[0];

	d = float2double(fi.f);

	return fi.f;
}