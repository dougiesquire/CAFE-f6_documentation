#!/bin/bash -l

# out_file_1="/scratch/ux06/ds0092/CAFE/forecasts/f6/WIP/c5-d60-pX-f6-19951101-reproducibility_test/mem001/mom-o1800-a1800-c1800.out"
# out_file_2="/g/data/xv83/users/ds0092/active_projects/Squire_2022_CAFE-f6/data/raw/c5-d60-pX-f6-19951101/c5-d60-pX-f6-19951101-mem001/scratch/ux06/ds0092/CAFE/forecasts/f6/WIP/c5-d60-pX-f6-19951101/mem001/mom-1800.out"

# out_file_1="/g/data/xv83/users/ds0092/active_projects/Squire_2022_CAFE-f6/data/raw/c5-d60-pX-f6-20080501/c5-d60-pX-f6-20080501-mem001/scratch/ux06/ds0092/CAFE/forecasts/f6/WIP/c5-d60-pX-f6-20080501/mem001/mom-1800.out"
# out_file_2="/scratch/ux06/ds0092/CAFE/forecasts/f6/WIP/c5-d60-pX-f6-20080501-reproducibility_test/mem001/mom-o1800-a1800-c1800.out"

out_file_1="/g/data/xv83/users/ds0092/active_projects/Squire_2022_CAFE-f6/data/raw/c5-d60-pX-f6-20090501/c5-d60-pX-f6-20090501-mem001/scratch/ux06/ds0092/CAFE/forecasts/f6/WIP/c5-d60-pX-f6-20090501/mem001/mom-1200.out"
out_file_2="/scratch/ux06/ds0092/CAFE/forecasts/f6/WIP/c5-d60-pX-f6-20090501-reproducibility_test/mem001/mom-o1200-a1200-c1200.out"

start=" ==================Summary of completed MOM integration======================="
stop="  ===================================================="
sed -n "/^${start}/,/^${stop}/{p;/^${stop}/q}" $out_file_1 > mom_summary_1.txt
sed -i '1s/^/'"${out_file_1//\//\\/}"'\n/' mom_summary_1.txt

sed -n "/^${start}/,/^${stop}/{p;/^${stop}/q}" $out_file_2 > mom_summary_2.txt
sed -i '1s/^/'"${out_file_2//\//\\/}"'\n/' mom_summary_2.txt

echo "`diff mom_summary_1.txt mom_summary_2.txt`"
