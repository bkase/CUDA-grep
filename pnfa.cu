
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "pnfa.h"

#define PRINT(time,...) if(!time) printf(__VA_ARGS__)

#define IS_EMPTY(l) (l->n == 0)
#define PUSH(l, state) l->s[l->n++] = state
#define POP(l) l->s[l->n]; 



__device__ inline void paddstate(List*, State*, List*, int *);
__device__ inline void pstep(List*, int, List*, int *);

/* Compute initial state list */
__device__ inline List*
pstartlist(State *start, List *l, int *dlistid)
{
	l->n = 0;
	(*dlistid)++;

	List addStartState;
	paddstate(l, start, &addStartState, dlistid);
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
paddstate(List *l, State *s, List *addStateList, int *dlistid)
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
		if(s == NULL);
		else if (s->c == 257) {
			s->lastlist = *dlistid; 
			PUSH(addStateList, s->out);
			PUSH(addStateList, s->out1);	
		}
		else {
			s->lastlist = *dlistid; 
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
pstep(List *clist, int c, List *nlist, int *dlistid)
{
	int i;
	State *s;
	(*dlistid)++;
	nlist->n = 0;
	for(i=0; i<clist->n; i++){
		s = clist->s[i];
	
		if(s->c == c || s->c == Any){
			List addStartState;
			paddstate(nlist, s->out, &addStartState, dlistid);
		}
	}
}

/* Run NFA to determine whether it matches s. */
__device__ inline int
pmatch(State *start, char *s, List *dl1, List *dl2, int * dlistid)
{
	int c;
	List *clist, *nlist, *t;

	clist = pstartlist(start, dl1, dlistid);
	nlist = dl2;
	for(; *s; s++){
		c = *s & 0xFF;
		pstep(clist, c, nlist, dlistid);
		t = clist; clist = nlist; nlist = t;	// swap clist, nlist 
	
		// check for a match in the middle of the string
		if (ispmatch(clist))
			return 1;

	}
	return ispmatch(clist);
}

/* Check for a string match at all possible start positions */
__device__ inline int panypmatch(State *start, char *s, List *dl1, List *dl2, int *dlistid) { 
	int isMatch = pmatch(start, s, dl1, dl2, dlistid);
	int index = 0;
	int len = 0; 
	char * sc = s;
	while(*sc != 0) {
		len ++;
		sc += 1;	
	}
	
	while (!isMatch && index < len) {
		isMatch = pmatch(start, s + index, dl1, dl2, dlistid);
		index ++;
	}
	return isMatch;
}


/* Allocate and initialize State */
__device__ inline State* pstate(int c, State *out, State *out1)
{
	State *s = (states + pnstate); // assign a state
	
	s->id = pnstate++;
	s->lastlist = 0;
	s->c = c;
	s->out = out;
	s->out1 = out1;
	
	// device pointer of itself
	// serves no real purpose other than to help transfer the NFA over
	s->dev = NULL;
	
	s->free = 0;
	return s;
}


/* Initialize frag struct. */
__device__ inline Frag
pfrag(State *start, Ptrlist *out)
{
	Frag n = { start, out };
	return n;
}

/* Create singleton list containing just outp. */
__device__ inline Ptrlist*
plist1(State **outp)
{
	Ptrlist *l;

	l = (Ptrlist*)outp;
	l->next = NULL;
	return l;
}

/* Patch the list of states at out to point to start. */
__device__ inline void
ppatch(Ptrlist *l, State *s)
{
	Ptrlist *next;

	for(; l; l=next){
		next = l->next;
		l->s = s;
	}
}

/* Join the two lists l1 and l2, returning the combination. */
__device__ inline Ptrlist*
pappend(Ptrlist *l1, Ptrlist *l2)
{
	Ptrlist *oldl1;

	oldl1 = l1;
	while(l1->next)
		l1 = l1->next;
	l1->next = l2;
	return oldl1;
}


/*
 * Convert postfix regular expression to NFA.
 * Return start state.
 */
 
__device__ inline State*
ppost2nfa(char *postfix)
{
	char *p;
	Frag stack[1000], *stackp, e1, e2, e;
	State *s;

	// fprintf(stderr, "postfix: %s\n", postfix);

	if(postfix == NULL)
		return NULL;

#define push(s) *stackp++ = s
#define pop() *--stackp

	stackp = stack;
	for(p=postfix; *p; p++){
		switch(*p){
            case ANY: /* any (.) */
				s = pstate(Any, NULL, NULL);
				push(pfrag(s, plist1(&s->out)));
				break;
			default:
				s = pstate(*p, NULL, NULL);
				push(pfrag(s, plist1(&s->out)));
				break;
			case CONCATENATE:	/* catenate */
				e2 = pop();
				e1 = pop();
				ppatch(e1.out, e2.start);
				push(pfrag(e1.start, e2.out));
				break;
			case ALTERNATE:	/* alternate (|)*/
				e2 = pop();
				e1 = pop();
				s = pstate(Split, e1.start, e2.start);
				push(pfrag(s, pappend(e1.out, e2.out)));
				break;
			case QUESTION:	/* zero or one (?)*/
				e = pop();
				s = pstate(Split, e.start, NULL);
				push(pfrag(s, pappend(e.out, plist1(&s->out1))));
				break;
			case STAR:	/* zero or more (*)*/
				e = pop();
				s = pstate(Split, e.start, NULL);
				ppatch(e.out, s);
				push(pfrag(s, plist1(&s->out1)));
				break;
			case PLUS:	/* one or more (+)*/
				e = pop();
				s = pstate(Split, e.start, NULL);
				ppatch(e.out, s);
				push(pfrag(e.start, plist1(&s->out1)));
				break;
		}
	}

	e = pop();
	if(stackp != stack)
		return NULL;

	ppatch(e.out, &pmatchstate);

	return e.start;
#undef pop
#undef push
}






__global__ void parallelMatch(char * bigLine, u32 * tableOfLineStarts, int lineIndex, int nstate, int time, char *postfix, unsigned char * devResult) {

	State s[100];
	pnstate = 0;
	states = s;

	State *st = ppost2nfa(postfix);

	List d1;
	List d2;	
	int dlistid;


	int i;
	for (i = blockIdx.x * blockDim.x + threadIdx.x; i < lineIndex; i += gridDim.x * blockDim.x) { 
       
        char * lineSegment = bigLine + tableOfLineStarts[i];

        if (panypmatch(st, lineSegment, &d1, &d2, &dlistid)) 
			devResult[i] = 1;
		else
			devResult[i] = 0;
	}
}

void pMatch(char * bigLine, u32 * tableOfLineStarts, int lineIndex, int nstate, int time, char * postfix, char **lines) {

	unsigned char *devResult;
	cudaMalloc(&devResult, lineIndex * sizeof(unsigned char));
	
	parallelMatch<<<256, 256>>>(bigLine, tableOfLineStarts, lineIndex, nstate ,time, postfix, devResult);
	
	cudaThreadSynchronize();

	//TODO free states
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

	unsigned char *hostResult = (unsigned char *) malloc (lineIndex * sizeof(unsigned char));
	cudaMemcpy(hostResult, devResult, lineIndex * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	for (int i = 0; i < lineIndex; i++) {
		if(hostResult[i] == 1) 
			printf("%s", lines[i]);
	}

	cudaFree(&bigLine);
    cudaFree(&tableOfLineStarts);

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
