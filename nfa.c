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

#include "pnfa.h"
#include "cycleTimer.h"

int
main(int argc, char **argv)
{	
	int visualize, simplified, postfix, i, time, parallel = 1;
	char *fileName = NULL;
	char *post;
    SimpleReBuilder builder;
	State *start;
	double startTime, endTime, endReadFile, endCopyStringsToDevice, endPMatch; 
	char **lines;
	int numLines;

	parseCmdLine(argc, argv, &visualize, &postfix, &fileName, &time, &simplified);

	// argv index at which regex is present
	int optIndex = 1 + visualize + postfix + time + simplified;
	if (fileName != NULL)
		optIndex += 2;

	if (argc <= optIndex) {
		usage (argv[0]);
		exit(EXIT_SUCCESS);
	}

    simplifyRe(argv[optIndex], &builder);

    post = re2post(builder.re);

    if (simplified == 1) {
        char * clean_simplified = stringify(builder.re);
        printf("\nSimplified Regex: %s\n", clean_simplified);
        free(clean_simplified);
        exit(0);
    }

    /* destruct the simpleRe */
    _simpleReBuilder(&builder);

	if(post == NULL){
		fprintf(stderr, "bad regexp %s\n", argv[optIndex]);
		return 1;
	}

    if (postfix == 1) {
        char * clean_post = stringify(post);
		printf("\nPostfix buffer: %s\n", clean_post);
        free(clean_post);
        exit(0);
	}

	if (visualize == 1) { 
		start = post2nfa(post);
		visualize_nfa(start);
        exit(0);
	}

	// sequential matching
	if (parallel != 1) {
		
		start = post2nfa(post);
		if(start == NULL){
			fprintf(stderr, "error in post2nfa %s\n", post);
		return 1;
		}

		// if no file is specified
		if (fileName == NULL) {
            startTime = CycleTimer::currentSeconds();
			for(i=optIndex+1; i<argc; i++) {
				if(anyMatch(start, argv[i]))
					printf("%d: %s\n", i-(optIndex), argv[i]);
			}
            endTime = CycleTimer::currentSeconds();
		}
		else {
			startTime = CycleTimer::currentSeconds();
		
			readFile(fileName, &lines, &numLines); 	

            unsigned char result[numLines];

			for (i = 0; i < numLines; i++) { 
				if (anyMatch(start, lines[i]))  
					result[i] = 1;
				else
					result[i] = 0;
			}
            endTime = CycleTimer::currentSeconds();
	
			for ( i = 0; i < numLines; i++) {
				if(result[i] == 1)
					printf("%s", lines[i]);
			}

			for (i = 0; i <= numLines; i++) 
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
		
	    simplifyRe(argv[optIndex], &builder);
	
		char *device_regex;
		int postsize = (strlen(builder.re) + 1) * sizeof (char);
		cudaMalloc((void **) &device_regex, postsize); 
		cudaMemcpy(device_regex, builder.re, postsize, cudaMemcpyHostToDevice);	
	
		startTime = CycleTimer::currentSeconds();	
		readFile(fileName, &lines, &numLines); 	 
    	endReadFile = CycleTimer::currentSeconds();

		char * device_line;
        u32 * device_table;
		copyStringsToDevice(lines, numLines, &device_line, &device_table);
        endCopyStringsToDevice = CycleTimer::currentSeconds();

		pMatch(device_line, device_table, numLines, time, device_regex, lines);
        endPMatch = CycleTimer::currentSeconds();

		for (i = 0; i <= numLines; i++) 
			free(lines[i]);
		free(lines);
	}

	if (time && !parallel) {
		printf("\nSequential Time taken %.4f \n\n", (endTime - startTime));
	}
    else if (time && parallel) {
		printf("\nParallel ReadFile Time taken %.4f \n", (endReadFile - startTime));
		printf("\nParallel CopyStringsToDevice Time taken %.4f \n", (endCopyStringsToDevice - endReadFile));
		printf("\nParallel pMatch Time taken %.4f \n\n", (endPMatch - endCopyStringsToDevice));
		printf("\nParallel Total Time taken %.4f \n\n", (endPMatch - startTime));
    }

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
