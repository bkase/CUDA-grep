#include "nfautil.h"


/*
 * Visualize the NFA in stdout
 */
int visited[5000];
int count[5000];
int visited_index = 0;


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
    else {
        data = malloc(sizeof(char)*10);
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

