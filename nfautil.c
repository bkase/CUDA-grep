#include "nfautil.h"

/*
 * Visualize the NFA in stdout
 */
int visited[5000];
int count[5000];
int visited_index = 0;

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
		case 0x05: // (
			if(natom > 1){
				--natom;
				*dst++ = 0x1b;
			}
			if(p >= paren+100)
				return NULL;
			p->nalt = nalt;
			p->natom = natom;
			p++;
			nalt = 0;
			natom = 0;
			break;
		case 0x04: // |
			if(natom == 0)
				return NULL;
			while(--natom > 0)
				*dst++ = 0x1b;
			nalt++;
			break;
		case 0x06: // )
			if(p == paren)
				return NULL;
			if(natom == 0)
				return NULL;
			while(--natom > 0)
				*dst++ = 0x1b;
			for(; nalt > 0; nalt--)
				*dst++ = 0x04;
			--p;
			nalt = p->nalt;
			natom = p->natom;
			natom++;
			break;
		case 0x03: // *
		case 0x01: // +
		case 0x02: // ?
			if(natom == 0)
				return NULL;
			*dst++ = *re;
			break;
		default:
			if(natom > 1){
				--natom;
				*dst++ = 0x1b;
			}
			*dst++ = *re;
			natom++;
			break;
		}
	}
	if(p != paren)
		return NULL;
	while(--natom > 0)
		*dst++ = 0x1b;
	for(; nalt > 0; nalt--)
		*dst++ = 0x04;
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
	printf("  -t	Print timing data\n");
    printf("  -f <FILE> --file Input file\n");	
	printf("  -? This message\n");
}

void parseCmdLine(int argc, char **argv, int *visualize, int *postfix, char **fileName, int *time) {
	if (argc < 3) {
		usage(argv[0]);
		exit(EXIT_SUCCESS);
	}
	
	int opt;
	static struct option long_options[] = {
        {"help",     no_argument, 0,  '?'},
        {"postfix",     no_argument, 0,  'p'}, 
		{"visualize",    no_argument, 0,  'v'},
		{"file",     required_argument, 0,  'f'},
		{"time",     no_argument, 0,  't'},
		{0 ,0, 0, 0}
    };

	*visualize = 0;
	*postfix = 0;
	*time = 0;
    while ((opt = getopt_long_only(argc, argv, "t:v:p:f:?", long_options, NULL)) != EOF) {

        switch (opt) {
        case 'v': {
			*visualize = 1;
			break;
		}
		case 'p': {
			*postfix = 1;	
			break;
		}
		case 'f': {
			*fileName = optarg; 
			break;
		}
		case 't': {
			*time = 1;
			break;
		}
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

