#!/bin/bash

#PBS -P xv83
#PBS -l storage=gdata/xv83
#PBS -q copyq
#PBS -l walltime=10:00:00
#PBS -l mem=100gb
#PBS -l wd
#PBS -j oe

years=( $(seq 1981 1 2020))
months=( 05 11 )

for year in ${years[@]}; do
	for month in ${months[@]}; do
		forecast="c5-d60-pX-f6-${year}${month}01"
		
		mkdir -p $forecast

		# mdss -P v14 dmls -l f6/${forecast}/${forecast}-base.tar > /dev/null 2>&1
		if [ $? -eq 0 ]; then
			project="v14"
			file="${forecast}-base.tar"	
		else
			# mdss -P v14 dmls -l f6/${forecast}/f6.WIP.${forecast}.top_level.*.tar > /dev/null 2>&1	
			if [ $? -eq 0 ]; then
				project="v14"
				file="f6.WIP.${forecast}.top_level.*.tar"
			else
				project="xv83"
				file="${forecast}-base.tar"
			fi
		fi
		# mdss -P ${project} get f6/${forecast}/${file} ./$forecast/
		
		mkdir -p ${forecast}/${forecast}-base
		# tar -xf ./${forecast}/${file} -C ./${forecast}/${forecast}-base
		
		# Delete everything except for fms_CM2M.x
		find ${forecast}/${forecast}-base ! -name 'fms_CM2M.x' -type f -exec rm -f {} +
		find ${forecast}/${forecast}-base -empty -type d -delete
		rm -rf ${forecast}/${file}
	done
done
