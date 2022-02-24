#!/bin/bash

#PBS -P xv83
#PBS -l storage=gdata/xv83
#PBS -q copyq
#PBS -l walltime=10:00:00
#PBS -l mem=100gb
#PBS -l wd
#PBS -j oe

years=( 2020 ) #( $(seq 1981 1 2020))
months=( 11 ) #( 05 11 )

for year in ${years[@]}; do
	for month in ${months[@]}; do
		forecast="c5-d60-pX-f6-${year}${month}01-reproducibility_test"
		
		mkdir -p $forecast

		mdss -P xv83 dmls -l f6/${forecast}/${forecast}-mem001.tar > /dev/null 2>&1
		if [ $? -eq 0 ]; then
			mdss -P xv83 get f6/${forecast}/${forecast}-mem001.tar ./$forecast/
		
			mkdir -p ${forecast}/${forecast}-mem001
			tar -xf ./${forecast}/${forecast}-mem001.tar -C ./${forecast}/${forecast}-mem001
		
			# Delete everything except for fms_CM2M.x
			find ${forecast}/${forecast}-mem001 ! -name 'ocean_scalar_*.nc' -type f -exec rm -f {} +
			find ${forecast}/${forecast}-mem001 -empty -type d -delete
			rm -rf ${forecast}/${file}
		fi
	done
done
