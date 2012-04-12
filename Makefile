CC=gcc
NVCC=nvcc
CFLAGS=-O3
CFLAGS_DEBUG=-g

all: nfa nfa_debug

.PHONY: clean

clean: 
	rm -rf nfa nfa_debug *.o

nfa: nfa.c nfautil.c 
	$(NVCC) $(CFLAGS) nfa.c nfautil.c -o nfa

nfa_debug: nfa.c nfautil.c
	$(NVCC) $(CFLAGS_DEBUG) nfa.c nfautil.c -o nfa_debug
