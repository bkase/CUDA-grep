
#include "nfautil.h"
#include "regex.h"

#ifndef PNFA_H
#define PNFA_H

typedef unsigned int u32;

void pMatch(State *start, char * bigLine, u32 * tableOfLineStarts, int lineIndex, int nstate, int time);
void printCudaInfo();
    
#endif
