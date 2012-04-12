#!/bin/bash

./nfa $1 $2 | sed s/'},]'/'}]'/ > srv/nfa.json
curl -F "data=@srv/nfa.json" http://128.237.92.16:3000/upload-nfa

