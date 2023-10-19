#!/bin/bash

start=$1  # Change this to your desired starting integer
end=$2    # Change this to your desired ending integer

for ((i=start; i<=end; i++)); do
    ./join_one_day.sh $i
done
