#!/bin/bash


wget https://www.climatechangeinaustralia.gov.au/media/ccia/2.2/cms_page_media/238/NRM_super_clusters.zip -P ../data/raw/NRM_super_clusters/

unzip ../data/raw/NRM_super_clusters/NRM_super_clusters.zip -d ../data/raw/NRM_super_clusters/
rm -rf ../data/raw/NRM_super_clusters/NRM_super_clusters.zip
