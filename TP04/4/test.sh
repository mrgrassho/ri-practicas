#!/usr/local/bin/bash

# Using: 
# GNU bash, version 5.0.11(1)-release (x86_64-apple-darwin18.6.0)

for number in {520..1000..20}
do
    echo "$number"
    echo -e "\n$number\n" >> TEST_RESULTS.md
    ./main.py -s ../stopword-list.txt -d ../en/ -t porter -b $number --not-verbose >> TEST_RESULTS.md
    rm -rf 4731/
done
exit 0