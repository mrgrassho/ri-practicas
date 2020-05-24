#!/usr/local/bin/bash

# Using: 
# GNU bash, version 5.0.11(1)-release (x86_64-apple-darwin18.6.0)

# Test all metrics with all the weights schemes

declare -a vl=("V1" "V2" "V3")
declare -a ml=("scalar_prod" "cosine")

for v in "${vl[@]}"
do
    for m in "${ml[@]}"
    do
        ./main.py -s ../stopword-list.txt -d ../en/ -t porter -q ../queries_2y3t.txt --weight $v --metric $m --not-verbose
        sleep(10)
    done
done
exit 0