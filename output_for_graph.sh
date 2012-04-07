#!/bin/bash

./a.out $1 $2 | sed s/'},]'/'}]'/ > srv/nfa.json

