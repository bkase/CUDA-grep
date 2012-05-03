#include "putil.cu" 

__device__ inline void paddstate(List*, State*, List*, int *);
__device__ inline void pstep(List*, int, List*, int *);


__device__ char buf[8000];

__device__ inline int pstrlen(char *str) {
	int len = 0; 
	while(*str != 0) {
		len ++;
		str += 1;
	}
	return len;
}

/*
 * Convert infix regexp re to postfix notation.
 * Insert ESC (or 0x1b) as explicit concatenation operator.
 * Cheesy parser, return static buffer.
 */
__device__ inline char * pre2post(char *re)
{
	int nalt, natom;
	char *dst;
	struct {
		int nalt;
		int natom;
	} paren[100], *p;
	
	p = paren;
	dst = buf;
	nalt = 0;
	natom = 0;
	
	int len = pstrlen(re);
	if(len >= sizeof buf/2)
		return NULL;
	for(; *re; re++){
		switch(*re){
		case PAREN_OPEN: // (
			if(natom > 1){
				--natom;
				*dst++ = CONCATENATE;
			}
			if(p >= paren+100)
				return NULL;
			p->nalt = nalt;
			p->natom = natom;
			p++;
			nalt = 0;
			natom = 0;
			break;
		case ALTERNATE: // |
			if(natom == 0)
				return NULL;
			while(--natom > 0)
				*dst++ = CONCATENATE;
			nalt++;
			break;
		case PAREN_CLOSE: // )
			if(p == paren)
				return NULL;
			if(natom == 0)
				return NULL;
			while(--natom > 0)
				*dst++ = CONCATENATE;
			for(; nalt > 0; nalt--)
				*dst++ = ALTERNATE;
			--p;
			nalt = p->nalt;
			natom = p->natom;
			natom++;
			break;
		case STAR: // *
		case PLUS: // +
		case QUESTION: // ?
			if(natom == 0)
				return NULL;
			*dst++ = *re;
			break;
		default:
			if(natom > 1){
				--natom;
				*dst++ = CONCATENATE;
			}	
			*dst++ = *re;
			natom++;
			break;
		}
	}
	if(p != paren)
		return NULL;
	while(--natom > 0)
		*dst++ = CONCATENATE;
	for(; nalt > 0; nalt--)
		*dst++ = ALTERNATE;
	*dst = 0;

	return dst;
}



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
		if(l->s[i]->c == Match)
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
	
		s = POP(addStateList);
	
		// lastlist check is present to ensure that if
		// multiple states point to this state, then only
		//one instance of the state is added to the list
		if(s == NULL);
		else if (s->c == Split) {
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
	int len = pstrlen(s);	
	while (!isMatch && index < len) {
		isMatch = pmatch(start, s + index, dl1, dl2, dlistid);
		index ++;
	}
	return isMatch;
}

__global__ void parallelMatch(char * bigLine, u32 * tableOfLineStarts, int numLines, int time, char *regexLines, u32 *regexTable, unsigned char * devResult) {

	pre2post(regexLines);

	char *postfix = buf;
	
	State s[100];
	pnstate = 0;
	states = s;

	State *st = ppost2nfa(postfix);

	List d1;
	List d2;	
	int dlistid;


	int i;
	for (i = blockIdx.x * blockDim.x + threadIdx.x; i < numLines; i += gridDim.x * blockDim.x) { 
       
        char * lineSegment = bigLine + tableOfLineStarts[i];

        if (panypmatch(st, lineSegment, &d1, &d2, &dlistid)) 
			devResult[i] = 1;
		else
			devResult[i] = 0;
	}
}

void pMatch(char * bigLine, u32 * tableOfLineStarts, int numLines, int time, char * regexLines, u32 *regexTable, char **lines) {

	unsigned char *devResult;
	cudaMalloc(&devResult, numLines * sizeof(unsigned char));

	parallelMatch<<<256, 256>>>(bigLine, tableOfLineStarts, numLines, time, regexLines, regexTable, devResult);
	cudaThreadSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

	unsigned char *hostResult = (unsigned char *) malloc (numLines * sizeof(unsigned char));
	cudaMemcpy(hostResult, devResult, numLines * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	for (int i = 0; i < numLines; i++) {
		if(hostResult[i] == 1) 
			PRINT(time, "%s", lines[i]);
		else 
			PRINT(time, "%s", lines[i]);
	}
	
	cudaFree(&devResult);
	cudaFree(&bigLine);
    cudaFree(&tableOfLineStarts);
}
