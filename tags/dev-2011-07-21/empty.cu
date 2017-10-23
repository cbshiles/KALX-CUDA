extern "C" {

__global__ void Empty(volatile unsigned int* const clockOut) 
{    
	volatile unsigned int clockValue = clock();    
	if(threadIdx.x == 0) 
	clockOut[blockIdx.x] = clock() - clockValue; 
}    

} // extern "C"

