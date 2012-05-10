#!/bin/bash

FILE=./sample

#if [[ $# -ne 2 ]]; then
#    echo "usage: $0 <regex> <file>"
#    exit 1
#fi

while read line
do
	egrep "^$line$" ./newlua &
	while `ps aux | grep 'egrep'` != ""
	do
		
		done

done < $FILE
