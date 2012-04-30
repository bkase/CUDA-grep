
#include "nfautil.h"
#include "regex.h"

typedef unsigned int u32;

void pMatch(State *start, char * bigLine, int * tableOfLineStarts, int lineIndex, int nstate, int time);
void printCudaInfo();
    
