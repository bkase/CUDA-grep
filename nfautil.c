#include "nfautil.h"

/*
 * Visualize the NFA in stdout
 */
int visited[5000];
int count[5000];
int visited_index = 0;

int nstate;
State matchstate = { Match };	/* matching state */

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
		if(l->s[i]->c == Match)
			return 1;
	return 0;
}

/* Add s to l, following unlabeled arrows. */
void
addstate(List *l, State *s)
{
	// lastlist check is present to ensure that if
	// multiple states point to this state, then only
	// one instance of the state is added to the list
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
copyStateToDevice(State **device_start, State *out, int pos) {

	if ((out != NULL)) { // && out->free != STATE_COPIED)) {
	
		State *device_out = out->dev;
		int done;
		// if the device version is not yet allocated, then allocate
		if (out->dev == NULL) {
			// allocate memory for out state & copy it over
			cudaMalloc((void **) &device_out, sizeof (State));	
			out->dev = device_out;
			cudaMemcpy(device_out, out, sizeof (State), cudaMemcpyHostToDevice);
			done = 0;
		}
		else {
			// if out->dev is not null, then we have already traversed out->out and out->out1 so stop
			done = 1;
		}
		
		// make start point to out
		if (pos == 0) 
			cudaMemcpy(&((*device_start)->out), &device_out, sizeof (State *), cudaMemcpyHostToDevice);
		else { 
			cudaMemcpy(&((*device_start)->out1), &device_out, sizeof (State *), cudaMemcpyHostToDevice);
		}

		if (done == 0) {
			copyStateToDevice(&device_out, out->out, 0);
			copyStateToDevice(&device_out, out->out1, 1);
		}
	}

}

void 
copyStringsToDevice(char **lines, int numLines, char ** device_line, u32 ** device_table) {
    
    //TODO: Is it more efficient to do this in two passes or to realloc a bunch of times
    //TODO: Instead of making some tableOfLineStarts empty use a different index for the avoided line[i]
    int size = 0;
    u32 * tableOfLineStarts = (u32 *)malloc(sizeof(u32)*(numLines+1));
    for (int i = 0; i < numLines; i++) {
        tableOfLineStarts[i] = size;
        size += strlen(lines[i]) + 1;
    }
    tableOfLineStarts[numLines] = size;

    char * bigLine = (char *)malloc(size);
    char * bigLineRunner = bigLine;
    for (int i = 0; i < numLines; i++) {
        //the size of this line is the runningTotal at the next index minus the runningTotal at this one (subtract off the null byte)
        int lineSize = (tableOfLineStarts[i+1] - tableOfLineStarts[i]) - 1;
        if (lineSize != 0) {
            memcpy(bigLineRunner, lines[i], lineSize);
            bigLineRunner[lineSize] = '\0';
            bigLineRunner += lineSize+1;
        }
    }
 
    cudaMalloc((void **) device_line, size);

    //TODO: check for cudaMalloc errors
    cudaMemcpy(*device_line, bigLine, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **) device_table, sizeof(u32)*(numLines+1));
    cudaMemcpy(*device_table, tableOfLineStarts, sizeof(u32)*(numLines+1), cudaMemcpyHostToDevice);

    free(tableOfLineStarts);
    free(bigLine);

}


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
	
	// device pointer of itself
	// serves no real purpose other than to help transfer the NFA over
	s->dev = NULL;
	
	s->free = STATE_INIT;
	return s;
}


/* Initialize Frag struct. */
	Frag
frag(State *start, Ptrlist *out)
{
	Frag n = { start, out };
	return n;
}


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
            case ANY: /* any (.) */
				s = state(Any, NULL, NULL);
				push(frag(s, list1(&s->out)));
				break;
			default:
				s = state(*p, NULL, NULL);
				push(frag(s, list1(&s->out)));
				break;
			case CONCATENATE:	/* catenate */
				e2 = pop();
				e1 = pop();
				patch(e1.out, e2.start);
				push(frag(e1.start, e2.out));
				break;
			case ALTERNATE:	/* alternate (|)*/
				e2 = pop();
				e1 = pop();
				s = state(Split, e1.start, e2.start);
				push(frag(s, append(e1.out, e2.out)));
				break;
			case QUESTION:	/* zero or one (?)*/
				e = pop();
				s = state(Split, e.start, NULL);
				push(frag(s, append(e.out, list1(&s->out1))));
				break;
			case STAR:	/* zero or more (*)*/
				e = pop();
				s = state(Split, e.start, NULL);
				patch(e.out, s);
				push(frag(s, list1(&s->out1)));
				break;
			case PLUS:	/* one or more (+)*/
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



/*
 * Convert infix regexp re to postfix notation.
 * Insert ESC (or 0x1b) as explicit concatenation operator.
 * Cheesy parser, return static buffer.
 */
char*
re2post(char *re)
{
	int nalt, natom;
	static char buf[8000];
	char *dst;
	struct {
		int nalt;
		int natom;
	} paren[100], *p;
	
	p = paren;
	dst = buf;
	nalt = 0;
	natom = 0;
	if(strlen(re) >= sizeof buf/2)
		return NULL;
	for(; *re; re++){
		switch(*re){
		case PAREN_OPEN: // (
			if(natom > 1){
				--natom;
				*dst++ = CONCATENATE;
			}
			if(p >= paren+100)
				return NULL;
			p->nalt = nalt;
			p->natom = natom;
			p++;
			nalt = 0;
			natom = 0;
			break;
		case ALTERNATE: // |
			if(natom == 0)
				return NULL;
			while(--natom > 0)
				*dst++ = CONCATENATE;
			nalt++;
			break;
		case PAREN_CLOSE: // )
			if(p == paren)
				return NULL;
			if(natom == 0)
				return NULL;
			while(--natom > 0)
				*dst++ = CONCATENATE;
			for(; nalt > 0; nalt--)
				*dst++ = ALTERNATE;
			--p;
			nalt = p->nalt;
			natom = p->natom;
			natom++;
			break;
		case STAR: // *
		case PLUS: // +
		case QUESTION: // ?
			if(natom == 0)
				return NULL;
			*dst++ = *re;
			break;
		default:
			if(natom > 1){
				--natom;
				*dst++ = CONCATENATE;
			}
			*dst++ = *re;
			natom++;
			break;
		}
	}
	if(p != paren)
		return NULL;
	while(--natom > 0)
		*dst++ = CONCATENATE;
	for(; nalt > 0; nalt--)
		*dst++ = ALTERNATE;
	*dst = 0;

	return buf;
}


void readFile(char *fileName, char ***lines, int *lineIndex) {
	
	FILE *fp = fopen(fileName, "r");
	if (fp == NULL) {
		printf("Error reading file \n");
		exit (EXIT_FAILURE);
	}

	int numLines = 8;
	// array of lines
	*lines = (char **)  malloc (sizeof(char *) * numLines); 
	
	// single line
	char *line = (char *) malloc (sizeof(char) * LINE_SIZE);
	
	*lineIndex = 0;
	while (fgets (line, LINE_SIZE, fp) != NULL) {
		(*lines)[(*lineIndex)] = line;	
		(*lineIndex) ++;

		line = (char *) malloc(sizeof(char) * LINE_SIZE);	

		if (*lineIndex == numLines-1) {
			numLines = numLines * 2;
			(*lines) = (char **) realloc((*lines), sizeof(char *) * numLines);
		}
	}
	(*lines)[(*lineIndex)] = line;	
	
	fclose(fp);
}


void usage(const char* progname) {
    printf("Usage: %s [options] [pattern] [text]*\n", progname);
    printf("Program Options:\n");
    printf("  -v	Visualize the NFA then exit\n");
    printf("  -p	View postfix expression then exit\n"); 
	printf("  -s	View simplified expression then exit\n");
	printf("  -t	Print timing data\n");
    printf("  -f <FILE> --file Input file\n");	
	printf("  -? This message\n");
}

void parseCmdLine(int argc, char **argv, int *visualize, int *postfix, char **fileName, int *time, int *simplified) {
	if (argc < 3) {
		usage(argv[0]);
		exit(EXIT_SUCCESS);
	}
	
	int opt;
	static struct option long_options[] = {
        {"help",     no_argument, 0,  '?'},
        {"postfix",     no_argument, 0,  'p'}, 
        {"simplified",     no_argument, 0,  's'}, 
		{"visualize",    no_argument, 0,  'v'},
		{"file",     required_argument, 0,  'f'},
		{"time",     no_argument, 0,  't'},
		{0 ,0, 0, 0}
    };

	*visualize = 0;
	*postfix = 0;
	*time = 0;
    *simplified = 0;
    while ((opt = getopt_long_only(argc, argv, "tvpsf:?", long_options, NULL)) != EOF) {

        switch (opt) {
            case 'v':
                *visualize = 1;
                break;

            case 'p':
                *postfix = 1;	
                break;

            case 'f':
                *fileName = optarg; 
                break;

            case 't':
                *time = 1;
                break;

            case 's':
                *simplified = 1;
                break;
                
            default: 
                usage(argv[0]);
                exit(EXIT_SUCCESS);
		} 
	}	

}


int hasSeen(State * start, int * index) {
    int i;
    for (i = 0; i < 5000; i++) {
        if (visited[i] == start->id) {
            *index = i;
            return 0;
        }
    }
    return 1;
}


void visualize_nfa_help(State * start) {
    int index;
    if (start == NULL) {
        return;
    }

    if (hasSeen(start, &index) == 0) {
        if (count[index] > 0) {
            return;
        }
    }

    count[start->id]++;
    visited[start->id] = start->id;
    
    char * data;
    if (start->c == Match) {
        data = "Match";
    }
    else if (start->c == Split) {
        data = "Split";
    }
    else if (start->c == Any) {
        data = "Any";
    }
    else {
        data = (char *) malloc(sizeof(char)*10);
        sprintf(data, "Char %c", start->c);
    }

    int outId, outId1;
    outId = (start->out == NULL) ? -1 : start->out->id;
    outId1 = (start->out1 == NULL) ? -1 : start->out1->id;

    printf("{ \"id\": \"%d\", \"data\":\"%s\", \"out\":\"%d\", \"out1\":\"%d\" \n},", start->id, data, outId, outId1);

    visualize_nfa_help(start->out);
    visualize_nfa_help(start->out1);
}

void visualize_nfa(State * start) {
    memset(visited, 0, 5000*(sizeof(int)));
    memset(count, 0, 5000*(sizeof(int)));
    printf("[");
    visualize_nfa_help(start);
    printf("]\n");
}

double gettime()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

