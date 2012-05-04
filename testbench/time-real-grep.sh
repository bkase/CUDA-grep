#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <regex> <file>"
    exit 1
fi

time egrep $1 $2
