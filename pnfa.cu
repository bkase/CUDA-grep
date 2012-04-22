
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "pnfa.h"

__device__ List *dl1, *dl2;
__device__ static int dlistid;
__device__ State pmatchstate = { Match };	/* matching state */


__device__ inline void paddstate(List*, State*);
__device__ inline void pstep(List*, int, List*);

/* Compute initial state list */
__device__ inline List*
pstartlist(State *start, List *l)
{
	l->n = 0;
	dlistid++;
	paddstate(l, start);
	return l;
}

/* Check whether state list contains a match. */
__device__ inline int
ispmatch(List *l)
{
	int i;

	for(i=0; i<l->n; i++)
		if(l->s[i] == &pmatchstate)
			return 1;
	return 0;
}

/* Add s to l, following unlabeled arrows. */
__device__ inline void
paddstate(List *l, State *s)
{
	// lastlist check is present to ensure that if
	// multiple states point to this state, then only
	// one instance of the state is added to the list
	if(s == NULL || s->lastlist == dlistid)
		return;
	s->lastlist = dlistid;
	if(s->c == Split){
		/* follow unlabeled arrows */
		paddstate(l, s->out);
		paddstate(l, s->out1);
		return;
	}
	l->s[l->n++] = s;
}

/*
 * pstep the NFA from the states in clist
 * past the character c,
 * to create next NFA state set nlist.
 */
__device__ inline void
pstep(List *clist, int c, List *nlist)
{
	int i;
	State *s;

	dlistid++;
	nlist->n = 0;
	for(i=0; i<clist->n; i++){
		s = clist->s[i];
		if(s->c == c || s->c == Any)
			paddstate(nlist, s->out);
	}
}

/* Run NFA to determine whether it matches s. */
__device__ inline int
pmatch(State *start, char *s)
{
	int c;
	List *clist, *nlist, *t;

	clist = pstartlist(start, dl1);
	nlist = dl2;
	for(; *s; s++){
		c = *s & 0xFF;
		pstep(clist, c, nlist);
		t = clist; clist = nlist; nlist = t;	// swap clist, nlist 

		// check for a match in the middle of the string
		if (ispmatch(clist))
			return 1;

	}
	return ispmatch(clist);
}

/* Check for a string match at all possible start positions */
__device__ inline int panypmatch(State *start, char *s) { 
	int isMatch = pmatch(start, s);
	int index = 0;
	int len = 0; 

	char * sc = s;
	while(sc != '\0') {
		len ++;
		sc += 1;
	}

	while (!isMatch && index <= len) {
		isMatch = pmatch(start, s + index);
		index ++;
	}
	return isMatch;
}


__global__ void parallelMatch(State *start, char **lines, int lineIndex, List* ddl1, List *ddl2) {
	printf("Entered kernel \n");
	dl1 = ddl1;
	dl2 = ddl2;

	int i;
	for (i = 0; i < lineIndex; i++) { 
	/*	if (panypmatch(start, lines[i])) 
			printf("%s", lines[i]);
	*/
	}
}

void pMatch(State *start, char **lines, int lineIndex, List* ddl1, List *ddl2) {
	//printCudaInfo(); 
	parallelMatch<<<1,1>>>(start, lines, lineIndex, ddl1, ddl2);	
}


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
