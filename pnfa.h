#ifndef PNFA_H
#define PNFA_H

#include "nfautil.h"
#include "regex.h"

typedef unsigned int u32;

__device__ State *states;
__device__ static int pnstate;
__device__ State pmatchstate = { Match };	/* matching state */

// host function which calls parallelNFAKernel
void parallelNFA(char *postfix);
// host function which calls parallelMatchingKernel
void pMatch(char * bigLine, u32 * tableOfLineStarts, int lineIndex, int nstate, int time, char *postfix, char **lines);
void printCudaInfo();
    
#endif
