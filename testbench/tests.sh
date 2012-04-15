#!/bin/bash

teststrings=(
"ROMEO" "JULIET"
#Test +*|
"ROMEO|JULIET" "R+" "R*" "R" "R+R*" "R*R+" "RR+" "RR*" "R+|J+" "(R|J)ULIET" 
#Test for the . wildcard
"But .hen" "R..EO" "R..EO|...IET" "R..*"
#Test for ranges
"[a-b]" "[q-s]" "[0-9]"
#Test for the \ sequence
"\?" "\|" "\+" "\*" "\." "\.\?\|\+\*" 
#Random tests
".*\?" "t*hi.\?"
);

date >> RESULTS;
echo "" >> RESULTS;
pass=1

len=${#teststrings[@]} 

for ((i=0; i<len; i++))
do
	testcase=${teststrings[i]};
	if diff <(../nfa -f ../romeojuliet.txt $i) <(egrep $i ../romeojuliet.txt) >> RESULTS; then
		cat /dev/null;
	else
		pass=0
		echo "Test Failed $testcase";
	fi
done

if diff <(../nfa -f ../romeojuliet.txt '.*') <(egrep '.*' ../romeojuliet.txt) >> RESULTS; then
		cat /dev/null;
else
	pass=0
	echo "Test Failed $testcase";
fi


if [ $pass -eq "1" ]; then
		echo "All tests passed" >> RESULTS;
fi
	
echo "------------" >> RESULTS;

