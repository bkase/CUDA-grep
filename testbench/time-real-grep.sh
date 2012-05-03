#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <regex> <file>"
    exit 1
fi

time grep $1 $2
