
#include "nfautil.h"
#include "regex.h"

__device__ State *states;
__device__ static int pnstate;
__device__ State pmatchstate = { Match };	/* matching state */

// host function which calls parallelNFAKernel
void parallelNFA(char *postfix);
// host function which calls parallelMatchingKernel
void pMatch(State *start, char **lines, int lineIndex, int nstate, int time, char *postfix);
void printCudaInfo();
    
