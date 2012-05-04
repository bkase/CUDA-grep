#ifndef NFAUTIL_H
#define NFAUTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <sys/time.h>

#define LINE_SIZE 200

#define ANY 0x15
#define CONCATENATE 0x1b
#define ALTERNATE 0x04
#define QUESTION 0x02
#define STAR 0x03
#define PLUS 0x01
#define PAREN_OPEN 0x05
#define PAREN_CLOSE 0x06

typedef struct State State;
struct State
{
	int c;
    int id;
	State *out;
	State *out1;
	
	State *dev;

	int lastlist;
	unsigned char free;
};


/*
 * Since the out pointers in the list are always 
 * uninitialized, we use the pointers themselves
 * as storage for the Ptrlists.
 */
typedef union Ptrlist Ptrlist;
union Ptrlist
{
	Ptrlist *next;
	State *s;
};


/*
 * A partially built NFA without the matching state filled in.
 * Frag.start points at the start state.
 * Frag.out is a list of places that need to be set to the
 * next state for this fragment.
 */
typedef struct Frag Frag;
struct Frag
{
	State *start;
	Ptrlist *out;
};

typedef struct List List;
struct List
{
	// only handle fixed number of states
	State *s[100];
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
void parseCmdLine(int argc, char **argv, int *visualize, int *postfix, int *time, int *simplified, char **fileName, char **regexFile); 
void visualize_nfa_help(State * start);
void visualize_nfa(State * start);
double gettime();


#define STATE_FREED 
#define STATE_INIT 0

#define DEBUG
#ifdef DEBUG
#define LOG(...) printf(__VA_ARGS__)
#endif

#ifndef DEBUG
#define LOG(...) //comment
#endif


typedef unsigned int u32;


State* state(int c, State *out, State *out1);
Frag frag(State *start, Ptrlist *out);
Ptrlist* list1(State **outp);
void patch(Ptrlist *l, State *s);
Ptrlist* append(Ptrlist *l1, Ptrlist *l2);
State* post2nfa(char *postfix);


List* startlist(State *start, List *l);
int ismatch(List *l);
void addstate(List *l, State *s);
void step(List *clist, int c, List *nlist);
int match(State *start, char *s);
int  anyMatch(State *start, char *s);
void copyStateToDevice(State **device_start, State *out, int pos);
void  copyStringsToDevice(char **lines, int numLines, char ** device_line, u32 ** device_table); 

#endif
