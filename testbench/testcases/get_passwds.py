#!/bin/env python

import sys

lastEmail = ""
for line in open(sys.argv[1]):
    sections = line.split('|')
    for i in xrange(len(sections)): 
        if '@' in sections[i]:
            if sections[i] != lastEmail:
                print sections[i] + " | " + sections[i-1]
            lastEmail = sections[i]
