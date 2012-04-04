#!/bin/bash

if [[ $1 == '' ]]
then
    gcc -Wall -O3 nfa.c -o a.out
elif [[ $1 == 'debug' ]]
then
    gcc -Wall -g3 nfa.c -o a.out
elif [[ $1 == 'clean' ]]
then
    rm -rf a.out
fi
