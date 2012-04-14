CC=gcc
NVCC=nvcc
CFLAGS=-O3
CFLAGS_DEBUG=-g

all: nfa nfa_debug

.PHONY: clean

clean: 
	rm -rf nfa nfa_debug *.o

nfa: pnfa.cu nfautil.c nfa.c
	$(NVCC) $(CFLAGS) pnfa.cu nfa.c nfautil.c -o nfa

nfa_debug: pnfa.cu nfa.c nfautil.c
	$(NVCC) $(CFLAGS_DEBUG) pnfa.cu nfa.c nfautil.c -o nfa_debug


