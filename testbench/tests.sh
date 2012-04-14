#!/bin/bash

teststrings=("ROMEO" "JULIET" "ROMEO|JULIET" "R+" "R*" "R" "R+R*" "R*R+" "RR+"
"R+|J+" "(R|J)ULIET" 
#Test for the . wildcard
"But .hen");

date >> RESULTS;
echo "" >> RESULTS;
pass=1

for i in ${teststrings[@]} 
do
	if diff <(../nfa -f ../romeojuliet.txt $i) <(egrep $i ../romeojuliet.txt) >> RESULTS; then
		cat /dev/null;
	else
		pass=0
		echo "Test Failed $i";
	fi
done

if [ $pass -eq "1" ]; then
		echo "All tests passed" >> RESULTS;
fi
	
echo "------------" >> RESULTS;

