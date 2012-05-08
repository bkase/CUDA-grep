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

char *post;

int checkCmdLine(int argc, char **argv, char **fileName, char **regexFile, int *time) {
	int visualize, simplified, postfix;
	SimpleReBuilder builder;
	State *start;

	parseCmdLine(argc, argv, &visualize, &postfix, time, &simplified, fileName, regexFile);

	// argv index at which regex is present
	int regexIndex = 1 + visualize + postfix + *time + simplified;
	if (fileName != NULL)
		regexIndex += 2;

	if (argc <= regexIndex) {
		usage (argv[0]);
		exit(EXIT_SUCCESS);
	}

    char * regexBuffer = (char*)malloc(strlen(argv[regexIndex])+1);
    strcpy(regexBuffer, argv[regexIndex]);
    simplifyRe(&regexBuffer, &builder);
    free(regexBuffer);

    post = re2post(builder.re);
	if(post == NULL){
		fprintf(stderr, "bad regexp %s\n", argv[regexIndex]);
		return 1;
	}

    if (simplified == 1) {
        char * clean_simplified = stringify(builder.re);
        printf("\nSimplified Regex: %s\n", clean_simplified);
        free(clean_simplified);
        exit(0);
    }

    /* destruct the simpleRe */
    _simpleReBuilder(&builder);

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

	return regexIndex;
	
}


int
main(int argc, char **argv)
{	
	int i, timerOn, parallel = 1;
	char *fileName = NULL, *regexFile = NULL, **lines = NULL, **regexs = NULL; 
	int numLines, numRegexs;
			
	SimpleReBuilder builder;
	State *start;
	double startTime, endTime, endReadFile, endCopyStringsToDevice, endPMatch; 

	int regexIndex = checkCmdLine(argc, argv, &fileName, &regexFile, &timerOn);
	
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
			for(i=regexIndex+1; i<argc; i++) {
				if(anyMatch(start, argv[i]))
					printf("%d: %s\n", i-(regexIndex), argv[i]);
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

		}

	}

	// parallel matching
	else {
		if (fileName == NULL) {
			printf("Enter a file \n");
			exit(EXIT_SUCCESS);
		}
		
		// match just a single regex
		if (regexFile == NULL) {
			char * regexBuffer = (char*)malloc(strlen(argv[regexIndex])+1);
            strcpy(regexBuffer, argv[regexIndex]);
			simplifyRe(&regexBuffer, &builder);
            free(regexBuffer);
	
			char *device_regex;
			int postsize = (strlen(builder.re) + 1) * sizeof (char);
			cudaMalloc((void **) &device_regex, postsize); 
			cudaMemcpy(device_regex, builder.re, postsize, cudaMemcpyHostToDevice);	
		
			startTime = CycleTimer::currentSeconds();	
			readFile(fileName, &lines, &numLines); 	 
			endReadFile = CycleTimer::currentSeconds();

			char * device_line;
			u32 * device_table;
			//copyStringsToDevice(lines, numLines, &device_line, &device_table);
			u32 * table = (u32 *) malloc(sizeof(u32) * strlen(*lines));
			table[0] = 0;
			int numLines = 0;
	
			int len = strlen(lines[0]);
			for (int i = 0; i < len; i++) {
				if ((lines[0])[i] == '\n') {
					table[++numLines] = i+1;
					lines[0][i] = 0;		
				
				}
			}
			-- numLines;	
	
			
			cudaMalloc(&device_table, sizeof (u32) * (len ));		
			cudaMalloc(&device_line, sizeof (char) * (len + 1));		

			cudaMemcpy(device_table, table, sizeof(u32) * (len), cudaMemcpyHostToDevice);
			cudaMemcpy(device_line, *lines, sizeof(char) * (len + 1), cudaMemcpyHostToDevice);
			
			endCopyStringsToDevice = CycleTimer::currentSeconds();

			u32 numRegexes = 1;
			pMatch(device_line, device_table, numLines, 1, timerOn, device_regex, &numRegexes, lines, table);
			endPMatch = CycleTimer::currentSeconds();
		}
		// match a bunch of regexs
		else {	
			startTime = CycleTimer::currentSeconds();	
			readFile(regexFile, &regexs, &numRegexs); 	 
			readFile(fileName, &lines, &numLines); 	 
			endReadFile = CycleTimer::currentSeconds();

	
			char * device_line;
			u32 * device_table;
			//copyStringsToDevice(lines, numLines, &device_line, &device_table);
			u32 * table = (u32 *) malloc(sizeof(u32) * strlen(*lines));
			table[0] = 0;
			int numLines = 0;
	
			int len = strlen(lines[0]);
			for (int i = 0; i < len; i++) {
				if ((lines[0])[i] == '\n') {
					table[++numLines] = i+1;
					lines[0][i] = 0;		
				
				}
			}
			-- numLines;	
	
				
			cudaMalloc(&device_table, sizeof (u32) * (len ));		
			cudaMalloc(&device_line, sizeof (char) * (len + 1));		

			cudaMemcpy(device_table, table, sizeof(u32) * (len), cudaMemcpyHostToDevice);
			cudaMemcpy(device_line, *lines, sizeof(char) * (len + 1), cudaMemcpyHostToDevice);
			
			char * device_regex;
			u32 * device_regex_table;
			u32 * host_regex_table = (u32 *) malloc(sizeof(u32) * strlen(*regexs));
			host_regex_table[0] = 0;
			int numRegexs = 0;
	

			//printf("REGEXS %s\n", regexs[0]);
			len = strlen(regexs[0]);
			for (int i = 0; i < len; i++) {
				if ((regexs[0])[i] == '\n') {
					host_regex_table[++numRegexs] = i+1;
					regexs[0][i] = 0;		
				
			/*	
            	    char * regexBuffer = (char*)malloc(strlen(regexs[host_regex_table[numRegexs-1]])+1);
               		 strcpy(regexBuffer, regexs[0][host_regex_table[numRegexs-1]]);
                	simplifyRe(&regexBuffer, &builder);
                	free(regexBuffer);

					regexs[0][host_regex_table[numRegexs-1]] = builder.re;
			
					printf("builder %s\n", builder.re);
			*/
				}
			}
			-- numRegexs;	
			/*	
			for (int i = 0; i < numRegexs; i++) {
				printf("%s\n", regexs[0] + host_regex_table[i]);
			}
			*/

			cudaMalloc(&device_regex_table, sizeof (u32) * (len ));		
			cudaMalloc(&device_regex, sizeof (char) * (len + 1));		

			cudaMemcpy(device_regex_table, host_regex_table, sizeof(u32) * (len), cudaMemcpyHostToDevice);
			cudaMemcpy(device_regex, *regexs, sizeof(char) * (len + 1), cudaMemcpyHostToDevice);
			
			endCopyStringsToDevice = CycleTimer::currentSeconds();

			pMatch(device_line, device_table, numLines, numRegexs, timerOn, device_regex, device_regex_table, lines, table);
			endPMatch = CycleTimer::currentSeconds();	
		
		}
	
	}

	
	// print timing details
	if (timerOn && !parallel) {
		printf("\nSequential Time taken %.4f \n\n", (endTime - startTime));
	}
    else if (timerOn && parallel) {
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
