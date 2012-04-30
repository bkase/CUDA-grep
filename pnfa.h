
#include "nfautil.h"
#include "regex.h"

// host function which calls parallelNFAKernel
void parallelNFA(char *postfix);
// host function which calls parallelMatchingKernel
void pMatch(State *start, char **lines, int lineIndex, int nstate, int time);
void printCudaInfo();
    
