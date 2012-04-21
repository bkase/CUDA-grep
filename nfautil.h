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
	int lastlist;
	unsigned char free;
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
void parseCmdLine(int argc, char **argv, int *visualize, int *postfix, char **fileName, int *time, int *simplified); 
void visualize_nfa_help(State * start);
void visualize_nfa(State * start);
double gettime();
