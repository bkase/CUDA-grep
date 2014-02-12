#ifndef PNFA_H
#define PNFA_H

#include "nfautil.h"
#include "regex.h"

#define PRINT(time,...) if(!time) printf(__VA_ARGS__)

#define IS_EMPTY(l) (l->n == 0)
#define PUSH(l, state) l->s[l->n++] = state
#define POP(l) l->s[--(l->n)]; 

// host function which calls parallelNFAKernel
void parallelNFA(char *postfix);
// host function which calls parallelMatchingKernel
void pMatch(char * bigLine, u32 * tableOfLineStarts, int numLines, int numRegexs, int time, char *regexLines, u32 *regexTable, char **lines, u32 *hostLineStarts);
 
#endif
