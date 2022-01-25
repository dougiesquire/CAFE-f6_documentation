#!/bin/bash

# get executables in chronological order
execs=( `ls -tr ../../data/raw/c5-d60-pX-f6-????????/c5-d60-pX-f6-????????-base/*/*/*/CAFE/forecasts/f6/WIP/c5-d60-pX-f6-????????/MOM/fms_CM2M.x` )


n_execs=${#execs[@]}
for i in $(seq 0 $(( $n_execs - 2 ))); do
        DIFF=$(diff ${execs[i]} ${execs[i+1]})
        forecast1=$(echo ${execs[i]} | grep -Eo c5-d60-pX-f6-........ | head -1)
	forecast2=$(echo ${execs[i+1]} | grep -Eo c5-d60-pX-f6-........ | head -1)
	date=`date -r ${execs[i+1]} '+%Y-%m-%d'`
	if [ "$DIFF" != "" ]; then
		date=`date -r ${execs[i+1]} '+%Y-%m-%d'`                
                # echo "The binary changed between ${forecast1} and ${forecast2} (${date})"
		echo -n "\\xmark & "
	else
		# echo "The binary is the same for ${forecast1} and ${forecast2}"
		echo -n "\\cmark & "
        fi
done
