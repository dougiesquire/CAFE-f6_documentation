#!/bin/bash

# This has to be run from the CSIRO side

variables="precip.nc tmax.nc tmin.nc"

PROJECT_DIR=/g/data/xv83/users/ds0092/active_projects/Squire_2022_CAFE-f6

for variable in ${variables}; do
	rsync -avPS /datasets/work/af-cdp/work/agcd/climate/${variable} ds0092@gadi-dm.nci.org.au:${PROJECT_DIR}/data/raw/AGCD/climate
	rsync -avPS /datasets/work/af-cdp/work/agcd/HISTORICAL/climate/${variable} ds0092@gadi-dm.nci.org.au:${PROJECT_DIR}/data/raw/AGCD/HISTORICAL/climate
done
