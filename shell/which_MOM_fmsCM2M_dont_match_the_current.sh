#!/bin/bash

CURRENT="../data/testing/reference_exectuable/fms_CM2M.x"

# Get forecast executables in chronological order
execs=( `ls -tr ../../data/testing/c5-d60-pX-f6-????????/c5-d60-pX-f6-????????-base/*/*/*/CAFE/forecasts/f6/WIP/c5-d60-pX-f6-????????/MOM/fms_CM2M.x` )

for exec in ${execs[@]}; do
	DIFF=$(diff ${CURRENT} ${exec})
	forecast=$(echo $exec | grep -Eo c5-d60-pX-f6-........ | head -1)
	# echo -n "`date -r ${exec} '+%Y-%m-%d'` & "
	if [ "$DIFF" != "" ]; then
		echo "${forecast} used a different execuable than the current one"
		# echo -n "\\xmark & "
	else
		echo "${forecast} used the same execuable as the current one"
		# echo -n "\\cmark & "	
	fi
done
