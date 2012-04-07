CC=gcc
CFLAGS=-O3 
CFLAGS_DEBUG=-g

all: wsp wsp_debug

.PHONY: clean

clean: 
	rm -rf wsp wsp_debug *~ *.o

wsp: nfa.c
	$(CC) $(CFLAGS) $< -o nfa

wsp_debug: nfa.c
	$(CC) $(CFLAGS_DEBUG) $< -o nfa_debug
