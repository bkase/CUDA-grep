#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>


/* Run NFA to determine whether it matches s. */

// TODO rewrite this to make simpler
/*
__device__ inline void
pmatch(State *start, char *s)
{
	int c;
	List *clist, *nlist, *t;

	clist = startlist(start, &l1);
	nlist = &l2;
	for(; *s; s++){
		c = *s & 0xFF;
		step(clist, c, nlist);
		t = clist; clist = nlist; nlist = t;	// swap clist, nlist 

		// check for a match in the middle of the string
		if (ismatch(clist))
			return 1;

	}
	return ismatch(clist);
}


// Check for a string match at all possible start positions
__device__ inline int
pAnyMatch(State *start, char *s) { 
	int isMatch = pmatch(start, s);
	int index = 0;
	int len = strlen(s);
	while (!isMatch && index <= len) {
		isMatch = pmatch(start, s + index);
		index ++;
	}
	return isMatch;
}

__global__ void
parallelMatch(int N, float alpha, float* x, float* y, float* result) {
	for (i = 0; i < lineIndex; i++) { 
			if (pAnyMatch(start, lines[i])) 
				//TODO need to put this statement out side loop body
				printf("%s", lines[i]);
		}
	}
}
*/


// taken from 15-418 assignment 2
void
printCudaInfo() {
    
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    printf("Found %d CUDA devices\n", deviceCount);
    
    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }

}
