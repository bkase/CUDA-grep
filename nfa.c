/*
 * Regular expression implementation.
 * Supports only ( | ) * + ?.  No escapes.
 * Compiles to NFA and then simulates NFA
 * using Thompson's algorithm.
 *
 * See also http://swtch.com/~rsc/regexp/ and
 * Thompson, Ken.  Regular Expression Search Algorithm,
 * Communications of the ACM 11(6) (June 1968), pp. 419-422.
 * 
 * Copyright (c) 2007 Russ Cox.
 * 
 * Can be distributed under the MIT license, see bottom of file.
 */

#include "nfautil.h"

#define DEBUG
#ifdef DEBUG
#define LOG(...) printf(__VA_ARGS__)
#endif

#ifndef DEBUG
#define LOG(...) //comment
#endif

State matchstate = { Match };	/* matching state */
int nstate;

/* Allocate and initialize State */
	State*
state(int c, State *out, State *out1)
{
	State *s;

	s = (State *) malloc(sizeof *s);
	s->id = ++nstate;
	s->lastlist = 0;
	s->c = c;
	s->out = out;
	s->out1 = out1;
	return s;
}

/*
 * A partially built NFA without the matching state filled in.
 * Frag.start points at the start state.
 * Frag.out is a list of places that need to be set to the
 * next state for this fragment.
 */
typedef struct Frag Frag;
typedef union Ptrlist Ptrlist;
struct Frag
{
	State *start;
	Ptrlist *out;
};

/* Initialize Frag struct. */
	Frag
frag(State *start, Ptrlist *out)
{
	Frag n = { start, out };
	return n;
}

/*
 * Since the out pointers in the list are always 
 * uninitialized, we use the pointers themselves
 * as storage for the Ptrlists.
 */
union Ptrlist
{
	Ptrlist *next;
	State *s;
};

/* Create singleton list containing just outp. */
	Ptrlist*
list1(State **outp)
{
	Ptrlist *l;

	l = (Ptrlist*)outp;
	l->next = NULL;
	return l;
}

/* Patch the list of states at out to point to start. */
	void
patch(Ptrlist *l, State *s)
{
	Ptrlist *next;

	for(; l; l=next){
		next = l->next;
		l->s = s;
	}
}

/* Join the two lists l1 and l2, returning the combination. */
	Ptrlist*
append(Ptrlist *l1, Ptrlist *l2)
{
	Ptrlist *oldl1;

	oldl1 = l1;
	while(l1->next)
		l1 = l1->next;
	l1->next = l2;
	return oldl1;
}



/*
 * Convert postfix regular expression to NFA.
 * Return start state.
 */
	State*
post2nfa(char *postfix)
{
	char *p;
	Frag stack[1000], *stackp, e1, e2, e;
	State *s;

	// fprintf(stderr, "postfix: %s\n", postfix);

	if(postfix == NULL)
		return NULL;

#define push(s) *stackp++ = s
#define pop() *--stackp

	stackp = stack;
	for(p=postfix; *p; p++){
		switch(*p){
            case '.':
				s = state(Any, NULL, NULL);
				push(frag(s, list1(&s->out)));
				break;
			default:
				s = state(*p, NULL, NULL);
				push(frag(s, list1(&s->out)));
				break;
			case 0x1b:	/* catenate */
				e2 = pop();
				e1 = pop();
				patch(e1.out, e2.start);
				push(frag(e1.start, e2.out));
				break;
			case '|':	/* alternate */
				e2 = pop();
				e1 = pop();
				s = state(Split, e1.start, e2.start);
				push(frag(s, append(e1.out, e2.out)));
				break;
			case '?':	/* zero or one */
				e = pop();
				s = state(Split, e.start, NULL);
				push(frag(s, append(e.out, list1(&s->out1))));
				break;
			case '*':	/* zero or more */
				e = pop();
				s = state(Split, e.start, NULL);
				patch(e.out, s);
				push(frag(s, list1(&s->out1)));
				break;
			case '+':	/* one or more */
				e = pop();
				s = state(Split, e.start, NULL);
				patch(e.out, s);
				push(frag(e.start, list1(&s->out1)));
				break;
		}
	}

	e = pop();
	if(stackp != stack)
		return NULL;

	patch(e.out, &matchstate);

	return e.start;
#undef pop
#undef push
}

typedef struct List List;
struct List
{
	State **s;
	int n;
};
List l1, l2;
static int listid;

void addstate(List*, State*);
void step(List*, int, List*);

/* Compute initial state list */
	List*
startlist(State *start, List *l)
{
	l->n = 0;
	listid++;
	addstate(l, start);
	return l;
}

/* Check whether state list contains a match. */
	int
ismatch(List *l)
{
	int i;

	for(i=0; i<l->n; i++)
		if(l->s[i] == &matchstate)
			return 1;
	return 0;
}

/* Add s to l, following unlabeled arrows. */
	void
addstate(List *l, State *s)
{
	if(s == NULL || s->lastlist == listid)
		return;
	s->lastlist = listid;
	if(s->c == Split){
		/* follow unlabeled arrows */
		addstate(l, s->out);
		addstate(l, s->out1);
		return;
	}
	l->s[l->n++] = s;
}

/*
 * Step the NFA from the states in clist
 * past the character c,
 * to create next NFA state set nlist.
 */
	void
step(List *clist, int c, List *nlist)
{
	int i;
	State *s;

	listid++;
	nlist->n = 0;
	for(i=0; i<clist->n; i++){
		s = clist->s[i];
		if(s->c == c || s->c == Any)
			addstate(nlist, s->out);
	}
}

/* Run NFA to determine whether it matches s. */
	int
match(State *start, char *s)
{
	int c;
	List *clist, *nlist, *t;

	clist = startlist(start, &l1);
	nlist = &l2;
	for(; *s; s++){
		c = *s & 0xFF;
		step(clist, c, nlist);
		t = clist; clist = nlist; nlist = t;	// swap clist, nlist 

		// check for a match in the middle of the string
		if (ismatch(clist))
			return 1;

	}
	return ismatch(clist);
}

/* Check for a string match at all possible start positions */
int 
anyMatch(State *start, char *s) { 
	int isMatch = match(start, s);
	int index = 0;
	int len = strlen(s);
	while (!isMatch && index <= len) {
		isMatch = match(start, s + index);
		index ++;
	}
	return isMatch;
}

/* device_start is a state that needs to have a 
 * pointer to out. This needs to be called for all states
 * Note: start has already been memcopyed over 
 * pos refers to whether its out or out1*/
void
copyStateToDevice(State *device_start, State *out, int pos) {

	if (out != NULL) {
		State *device_out;
		// allocate memory for out state & copy it over
		cudaMalloc((void **) &device_out, sizeof (State));	
		cudaMemcpy(&device_out, &out, sizeof (State), cudaMemcpyHostToDevice);
		// make start point to out
		if (pos == 0) 
			cudaMemcpy(&(device_start->out), &device_out, sizeof (State), cudaMemcpyHostToDevice);
		else 
			cudaMemcpy(&(device_start->out1), &device_out, sizeof (State), cudaMemcpyHostToDevice);
	
		copyStateToDevice(device_out, out->out, 0);
		copyStateToDevice(device_out, out->out1, 1);
	}	

}

void 
copyNFAToDevice(State **device_start, State *start) {
	cudaMalloc((void **) device_start, sizeof (State));
	cudaMemcpy(device_start, &start, sizeof (State), cudaMemcpyHostToDevice);
	copyStateToDevice(*device_start, start->out, 0);
	copyStateToDevice(*device_start, start->out1, 1);
}

void 
copyStringsToDevice(char **lines, int lineIndex, char ***device_lines) {

	// allocate memory for pointers
	cudaMalloc((void **) device_lines, sizeof (char *) * lineIndex);
	// copy each line over
	for (int i = 0; i < lineIndex; i++) {
		char *line; 
		cudaMalloc((void **) &line, sizeof (char) * LINE_SIZE);
		cudaMemcpy(line, lines[i], sizeof (char) * LINE_SIZE, cudaMemcpyHostToDevice);	
		cudaMemcpy(&((*device_lines)[i]), &line, sizeof (char *), cudaMemcpyDeviceToDevice); 	
	}


}

// free all states except Match which is statically allocated
void freeNFAStates(State *s) {
	if (s != NULL && s->c != Match) {
		freeNFAStates(s->out);
		freeNFAStates(s->out1);
		free(s);
	}
}

int
main(int argc, char **argv)
{	
	int visualize, postfix, i, time, parallel = 0;
	char *fileName = NULL;
	char *post;
	State *start;
	double starttime, endtime; 
	char **lines;
	int lineIndex;

	parseCmdLine(argc, argv, &visualize, &postfix, &fileName, &time);

	// argv index at which regex is present
	int optIndex = 1 + visualize + postfix + time;
	if (fileName != NULL)
		optIndex += 2;

	if (argc <= optIndex) {
		usage (argv[0]);
		exit(EXIT_SUCCESS);
	}

	post = re2post(argv[optIndex]);
	if(post == NULL){
		fprintf(stderr, "bad regexp %s\n", argv[optIndex]);
		return 1;
	}

    if (postfix == 1) {
		printf("\nPostfix buffer: %s\n", post);
        exit(0);
	}

	start = post2nfa(post);
	if(start == NULL){
		fprintf(stderr, "error in post2nfa %s\n", post);
		return 1;
	}

	if (visualize == 1) { 
		visualize_nfa(start);
        exit(0);
	}

	l1.s = (State **) malloc(nstate*sizeof l1.s[0]);
	l2.s = (State **) malloc(nstate*sizeof l2.s[0]);

	fflush(stdout); // flush stdout before getting start time

	// sequential matching
	if (parallel != 1) {

		// if no file is specified
		if (fileName == NULL) {
			starttime = gettime();
			for(i=optIndex+1; i<argc; i++) {
				if(anyMatch(start, argv[i]))
					printf("%d: %s\n", i-(optIndex), argv[i]);
			}
			endtime = gettime();
		}
		else {
			readFile(fileName, &lines, &lineIndex); 	

			starttime = gettime(); 
			for (i = 0; i < lineIndex; i++) { 
				if (anyMatch(start, lines[i])) 
					//TODO need to put this statement out side loop body
					printf("%s", lines[i]);
			}
			endtime = gettime();
		
			for (i = 0; i <= lineIndex; i++) 
			free(lines[i]);
			free(lines);
		}

	}

	// parallel matching
	else {
		if (fileName == NULL) {
			printf("Enter a file \n");
			exit(EXIT_SUCCESS);
		}
	
		readFile(fileName, &lines, &lineIndex); 	

		State *device_start;
		char **device_lines;
		copyNFAToDevice(&device_start, start);	
		copyStringsToDevice(lines, lineIndex, &device_lines);
	
		//TODO kernel call
		//parallelMatch<<<1,1>>>(device_start, device_lines, lineIndex);
		
		//TODO free up GPU memory	
	
		for (i = 0; i <= lineIndex; i++) 
		free(lines[i]);
		free(lines);

	}

	if (time) {
		printf("\nTime taken %f \n\n", (endtime - starttime));
	}
	// free up memory
	freeNFAStates(start);		

	free(l1.s);
	free(l2.s);

	return EXIT_SUCCESS;
}


/*
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the
 * Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall
 * be included in all copies or substantial portions of the
 * Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
 * KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS
 * OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
