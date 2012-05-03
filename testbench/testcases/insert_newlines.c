#include <stdio.h>
#include <string.h>

int main() {

    FILE * oneline = fopen("lua.js", "r");
    FILE * manylines = fopen("lua.lines.js", "w");

    char buf[100];
    while (fgets(buf, 100, oneline) != NULL) {
        fprintf(manylines, "%s\n", buf);
    }
    
    fclose(oneline);
    fclose(manylines);
}
