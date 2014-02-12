#ifndef PNFA_H
#define PNFA_H

#include "nfautil.h"
#include "regex.h"

#define PRINT(time,...) if(!time) printf(__VA_ARGS__)

#define IS_EMPTY(l) (l->n == 0)
#define PUSH(l, state) l->s[l->n++] = state
#define POP(l) l->s[--(l->n)]; 

__device__ State *states;
__device__ static int pnstate;
__device__ State pmatchstate = { Match };	/* matching state */


#endif
