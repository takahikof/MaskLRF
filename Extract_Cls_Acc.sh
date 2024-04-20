#!/bin/bash

find ./experiments -maxdepth 1 -mindepth 1 -type d | sort | while read i
do
    dir=$i
    logfile=`ls $dir/*.log`
    result=`tail -n 1 $logfile`
    result=($result) # convert to an array
    echo $dir "${result[11]}" # best classification accuracy
done
exit
