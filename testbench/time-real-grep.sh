#!/bin/bash

FILE=./sample

check_process() {
    [ "$1" = "" ] && return 0
    [ `pgrep -n $1` ] && return 1 || return 0
}


#if [[ $# -ne 2 ]]; then
#    echo "usage: $0 <regex> <file>"
#    exit 1
#fi

while read line
do
	egrep "^$line$" ./testcases/luaBig.50mb
done < $FILE



#while [ 1 ]; do
    #check_process "egrep"
    #[ $? -eq 0 ] && exit 1
    #sleep 0.1
#done
