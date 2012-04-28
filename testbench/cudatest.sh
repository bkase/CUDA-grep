#!/bin/bash

teststrings=(
"ROMEO" "JULIET"
"ROMEO|JULIET" "R." "R.MEO|JULIET"
);

date >> RESULTS
echo "" >> RESULTS
pass=1

len=${#teststrings[@]} 

FILES=./testcases/*

#loop over all test cases
for file in $FILES
do
	for ((i=0; i<len; i++))
	do
		testcase=${teststrings[$i]}
		if diff <(../nfa -f $file $testcase) <(egrep $testcase $file) >> RESULTS; then
			cat /dev/null
		else
			pass=0
			echo "Test Failed $testcase"
		fi
	done
done

if [ $pass -eq "1" ]; then
		echo "All tests passed" >> RESULTS
fi
	
echo "------------" >> RESULTS

