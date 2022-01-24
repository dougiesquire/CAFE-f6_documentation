#!/bin/bash

DIR1=/g/data/xv83/users/ds0092/data/CAFE/forecasts/f6/WIP/c5-d60-pX-f6-20060501-rerun-oldexec/c5-d60-pX-f6-20060501-rerun-oldexec-base/scratch/ux06/ds0092/CAFE/forecasts/f6/WIP/c5-d60-pX-f6-20060501-rerun-oldexec

DIR2=/g/data/xv83/users/ds0092/data/CAFE/forecasts/f6/WIP/c5-d60-pX-f6-20060501-rerun/c5-d60-pX-f6-20060501-rerun-base/scratch/ux06/ds0092/CAFE/forecasts/f6/WIP/c5-d60-pX-f6-20060501-rerun

for file in `find $DIR1 -type f`; do
	file_ext=${file#"$DIR1"}

	DIFF=$(diff ${DIR1}${file_ext} ${DIR2}${file_ext})
	if [ "$DIFF" != "" ]; then
        	echo "==================================================="
        	echo "The following things are different in $file_ext:"
       		echo "==================================================="
        	echo -e "$DIFF"
        	echo ""
	fi
done
