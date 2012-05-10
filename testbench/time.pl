#!/usr/bin/perl

refile = './sample';
#my DATAFILE = "testcases/luaBig.50mb";
datafile = './data';

while(<REFILE>) {
    re = $_;
    while(<DATAFILE>) {
        line = $_;
        if ($line =~ m/^$line$/) {
            print $line;
        }
    }
}

