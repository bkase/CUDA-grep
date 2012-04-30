
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "pnfa.h"

#define PRINT(time,...) if(!time) printf(__VA_ARGS__)

#define IS_EMPTY(l) (l->n == 0)
#define PUSH(l, state) l->s[l->n++] = state
#define POP(l) l->s[l->n]; 


__device__ static int dlistid;
__device__ State pmatchstate = { Match };	/* matching state */


__device__ inline void paddstate(List*, State*, List*);
__device__ inline void pstep(List*, int, List*);

/* Compute initial state list */
__device__ inline List*
pstartlist(State *start, List *l)
{
	l->n = 0;
	dlistid++;

	List addStartState;
	paddstate(l, start, &addStartState);
	return l;
}

/* Check whether state list contains a match. */
__device__ inline int
ispmatch(List *l)
{
	int i;

	for(i=0; i<l->n; i++) {
		if(l->s[i]->c == 256)
			return 1;
	}
	return 0;
}

/* Add s to l, following unlabeled arrows. */
	__device__ inline void
paddstate(List *l, State *s, List *addStateList)
{	
	addStateList->n = 0;
	PUSH(addStateList, s);
	/* follow unlabeled arrows */
	while(!IS_EMPTY(addStateList)) {	
	
		addStateList->n--;
		s = POP(addStateList);
	
		// lastlist check is present to ensure that if
		// multiple states point to this state, then only
		//one instance of the state is added to the list
		if(s == NULL || s->lastlist == dlistid);
		else if (s->c == 257) {
			s->lastlist = dlistid; 
			PUSH(addStateList, s->out);
			PUSH(addStateList, s->out1);	
		}
		else {
			s->lastlist = dlistid; 
			l->s[l->n++] = s;
		}
	}
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
	
		if(s->c == c || s->c == Any){
			List addStartState;
			paddstate(nlist, s->out, &addStartState);
		}
	}
}

/* Run NFA to determine whether it matches s. */
__device__ inline int
pmatch(State *start, char *s, List *dl1, List *dl2)
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
__device__ inline int panypmatch(State *start, char *s, List *dl1, List *dl2) { 
	int isMatch = pmatch(start, s, dl1, dl2);
	int index = 0;
	int len = 0; 
	char * sc = s;
	while(*sc != 0) {
		len ++;
		sc += 1;	
	}
	
	while (!isMatch && index < len) {
		isMatch = pmatch(start, s + index, dl1, dl2);
		index ++;
	}
	return isMatch;
}


__global__ void parallelMatch(State *start, char **lines, int lineIndex, int nstate, int time) {
	List d1;
	List d2;	


	int i;
	for (i = blockIdx.x * blockDim.x + threadIdx.x; i < lineIndex; i += gridDim.x * blockDim.x) { 
		if (panypmatch(start, lines[i], &d1, &d2)) 
			PRINT(time, "%s", lines[i]);
	}

/*
	// test to ensure that strings are copied over correctly
	for( int i = 0; i < lineIndex; i++) {
		printf("%s", lines[i]);		
	}
*/
	
}

void pMatch(State *start, char **lines, int lineIndex, int nstate, int time) {
		//printCudaInfo(); 
	parallelMatch<<<1,1>>>(start,lines,lineIndex, nstate ,time);


	//TODO free states

	int i;	
	for (i = 0; i <= lineIndex; i++) 
		cudaFree(&(lines[i]));
	cudaFree(&lines);

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
