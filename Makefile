CC=gcc
NVCC=nvcc
CFLAGS=-O3 -m64 -arch compute_20
CFLAGS_DEBUG=-g -m64 -arch compute_20


all: nfa nfa_debug

.PHONY: clean

clean: 
	rm -rf nfa nfa_debug *.o

nfa: pnfa.cu putil.cu nfautil.cpp nfa.cpp regex.cpp
	$(NVCC) $(CFLAGS) pnfa.cu putil.cu nfa.cpp nfautil.cpp regex.cpp -o nfa

nfa_debug: pnfa.cu putil.cu nfa.cpp nfautil.cpp regex.cpp
	$(NVCC) $(CFLAGS_DEBUG) pnfa.cu putil.cu nfa.cpp nfautil.cpp regex.cpp -o nfa_debug

