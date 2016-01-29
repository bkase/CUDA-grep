#ifndef PNFA_H
#define PNFA_H

#include "nfautil.h"
#include "regex.h"

#define PRINT(time,...) if(!time) printf(__VA_ARGS__)

#define IS_EMPTY(l) (l->n == 0)
#define PUSH(l, state) l->s[l->n++] = state
#define POP(l) l->s[--(l->n)]; 

__device__ __shared__ State *states; /*this variable used in ppost2nfa
                            it must be local for each block
                            simple way to do this use __shared__
                          */
__device__ __shared__  int pnstate; /*this variable too must be local
                                            for each blocks*/

__device__ State pmatchstate = { Match };	/* matching state */


#endif
