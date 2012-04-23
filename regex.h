#ifndef REGEX_H
#define REGEX_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nfautil.h"

typedef struct {
	int i;
	int size;
	char * re;
} SimpleReBuilder;

/* Constructor */
void simpleReBuilder(SimpleReBuilder ** builder, int len);

/* Destructor */
void _simpleReBuilder(SimpleReBuilder * builder);


void regex_error(int i);
char * stringify(char * nonull, int j);
void handle_escape(SimpleReBuilder * builder, char * complexRe, int len, int * bi, int * ci);
void putRange(SimpleReBuilder * builder, char start, char end, int * bi);
void handle_range(SimpleReBuilder * builder, char * complexRe, int len, int * bi, int * ci);
SimpleReBuilder * simplifyRe(char * complexRe, SimpleReBuilder * builder);
void freeNFAStates(State *s);
char * stringify(const char * oldRegex);

#endif
