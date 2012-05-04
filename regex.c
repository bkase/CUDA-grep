#include "nfautil.h"
#include "regex.h"

#define DEREF(arr,i) ((*(arr))[(i)])

/* constructor for SimpleReBuilder */
void simpleReBuilder(SimpleReBuilder ** builder, int len) {
    (*builder)->re = (char *)malloc(len+3);
    (*builder)->size = len+3;
}

void _simpleReBuilder(SimpleReBuilder * builder) {
    free(builder->re);
    /* the rest is on the stack */
}

void regex_error(int i) {
    fprintf(stderr, "Improper regex at character %d", i);
    exit(1);
}

char * stringify(char * nonull, int j) {
    char * proper = (char*)malloc(j+2);   
    memcpy(proper, nonull, j);
    proper[j+1] = '\0';
    return proper;
}

void handle_escape(SimpleReBuilder * builder, char ** complexRe, int * len, int * bi, int * ci) {

    int i = *ci;
    int j = *bi;
    int oldI;

    if (i+1 > *len)
        regex_error(i);

    oldI = ++i; //go to escaped character and ignore '/'
    switch(DEREF(complexRe,i)) {
        
        case 'd':
            char * buf;
            /* enough space for complexRe+the new range */
            *len = *len+6;
            *complexRe = (char*)realloc(*complexRe, *len);



            buf = (char*)malloc(*len);
            for (int k = i; k < *len-6; k++) {
                buf[k-i] = DEREF(complexRe,k);
            }
            DEREF(complexRe,i++) = '[';
            DEREF(complexRe,i++) = '0';
            DEREF(complexRe,i++) = '-';
            DEREF(complexRe,i++) = '9';
            DEREF(complexRe,i++) = ']';
            for (int k = i; k < *len; k++) {
                DEREF(complexRe,k) = buf[k-i+1];
            }

            i = oldI;
            
            free(buf);
            break;

        case 'w':
            printf("word character here\n");
            break;

        /* ... see www.cs.tut.fi/~jkorpela/perl/regexp.html */
        
        // default is just ignoring the backslash and taking the 
        // LITERAL character after no matter what
        default:
            builder->re[j++] = *complexRe[i++];
            break;
    }

    *ci = i-1; //-1 because we incremented at the end
    *bi = j-1; //-1 because we incremented at the end
}

void putRange(SimpleReBuilder * builder, char start, char end, int * bi) {
    
    int i = *bi;
    int amount = ((end - start + 1)*2)+1;
    builder->size += amount;
    builder->re = (char *)realloc(builder->re, builder->size);

    builder->re[i++] = PAREN_OPEN;
    builder->re[i++] = start;
    for (char k = start+1; k <= end; k++) {
        builder->re[i++] = ALTERNATE;
        builder->re[i++] = k;
    }
    builder->re[i] = PAREN_CLOSE;

    *bi = i;
}

void handle_range(SimpleReBuilder * builder, char * complexRe, int len, int * bi, int * ci) {
    
    int i = *ci;
    if (complexRe[i+4] != ']' || complexRe[i+2] != '-' || complexRe[i+1] > complexRe[i+3] || complexRe[i+1] <= 0x20) {
        fprintf(stderr, "Invalid range at character %d\n", i);
        exit(1);
    }

    putRange(builder, complexRe[i+1], complexRe[i+3], bi);
    *ci = i+4;
}

SimpleReBuilder * simplifyRe(char ** complexRe, SimpleReBuilder * builder) {

    int len = strlen(*complexRe);
    simpleReBuilder(&builder, len);

    int i,j;
    for (i = 0, j = 0; i < len; i++, j++) {
        switch(DEREF(complexRe, i)) {
            
            case '\\':
                handle_escape(builder, complexRe, &len, &j, &i);
                break;

            case '.':
                builder->re[j] = ANY; //nak is ANY
                break;

            case '+':
                builder->re[j] = PLUS; //0x01 is +
                break;

            case '?':
                builder->re[j] = QUESTION; //0x02 is ?
                break;

            case '*':
                builder->re[j] = STAR; //0x03 is *
                break;

            case '|':
                builder->re[j] = ALTERNATE; //0x04 is |
                break;

            case '(':
                builder->re[j] = PAREN_OPEN; //0x05 is (
                break;

            case ')':
                builder->re[j] = PAREN_CLOSE; //0x06 is )
                break;

            case '[':
                handle_range(builder, *complexRe, len, &j, &i);
                break;

            default:
                builder->re[j] = DEREF(complexRe,i);
                break;
        }

    }
    builder->re[j] = '\0';

    return builder;

}

char * stringify(const char * oldRegex) {
    int len = strlen(oldRegex);
    char * newRegex = (char*)malloc(len+1);
    int i;
    for (i = 0; i < len; i++) {
        switch(oldRegex[i]) {
            case ANY:
                newRegex[i] = '.';
                break;
            case CONCATENATE:
                newRegex[i] = '`';
                break;
            case ALTERNATE:
                newRegex[i] = '|';
                break;
            case QUESTION:
                newRegex[i] = '?';
                break;
            case STAR:
                newRegex[i] = '*';
                break;
            case PLUS:
                newRegex[i] = '+';
                break;
            case PAREN_OPEN:
                newRegex[i] = '(';
                break;
            case PAREN_CLOSE:
                newRegex[i] = ')';
                break;
            default:
                newRegex[i] = oldRegex[i];
                break;
        }
    }
    newRegex[i] = '\0';
    return newRegex;
}
