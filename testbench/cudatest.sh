#!/bin/bash

teststrings=(
"ROMEO" "JULIET"
#Test +*|
"ROMEO|JULIET" "R+" "R*" "R" "R+R*" "R*R+" "RR+" "RR*" "R+|J+" "(R|J)ULIET" 
#Test for the . wildcard
"R..EO" "R..EO|...IET" "R..*"
#Test for ranges
"[a-b]" "[q-s]" "[0-9]"
#Test for the \ sequence
"\?" "\|" "\+" "\*" "\." "\.\?\|\+\*" 
#Random tests
".*\?" "t*hi.\?"
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
		if diff <(sort <(../nfa -f $file $testcase)) <(sort <(egrep $testcase $file)) >> RESULTS; then
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

