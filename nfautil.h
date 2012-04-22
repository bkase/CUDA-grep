#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <sys/time.h>

#define LINE_SIZE 200

typedef struct State State;
struct State
{
	int c;
    int id;
	State *out;
	State *out1;
	int lastlist;
	unsigned char free;
};


typedef struct List List;
struct List
{
	State **s;
	int n;
};

/*
 * Represents an NFA state plus zero or one or two arrows exiting.
 * if c == Match, no arrows out; matching state.
 * If c == Split, unlabeled arrows to out and out1 (if != NULL).
 * If c == Any, unlabeled arrows to out (if != NULL).
 * If c < 256, labeled arrow with character c to out.
 */
enum
{
	Match = 256,
	Split = 257,
    Any   = 258
};


void readFile(char *fileName, char ***lines, int *lineIndex);
char* re2post(char *re);
void usage(const char* progname);
void parseCmdLine(int argc, char **argv, int *visualize, int *postfix, char **fileName, int *time); 
void visualize_nfa_help(State * start);
void visualize_nfa(State * start);
double gettime();
