#!/bin/bash
trap "exit" INT

ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"

#cd $ABSOLUTE_PATH../src && python cmp_start.py twitter_collector --log D &
##cd $ABSOLUTE_PATH../src && python cmp_start.py logger &
#cd $ABSOLUTE_PATH../src && python cmp_start.py preprocessor &
#cd $ABSOLUTE_PATH../src && python cmp_start.py mongo_insights --log D &
#cd $ABSOLUTE_PATH../src && python cmp_start.py mbti_classifier &
#####cd $ABSOLUTE_PATH../src && python cmp_start.py ibm_ocean &
#####cd $ABSOLUTE_PATH../src && python cmp_start.py ibm_vision &
#cd $ABSOLUTE_PATH../src && python cmp_start.py img_torch &
#####cd $ABSOLUTE_PATH../src && python cmp_start.py ibm_nlu --log D &
####cd $ABSOLUTE_PATH../src && python cmp_start.py age_classifier --log D &
###cd $ABSOLUTE_PATH../src && python cmp_start.py filter_db &
#cd $ABSOLUTE_PATH../src && python cmp_start.py filter_m3 &
#cd $ABSOLUTE_PATH../src && python cmp_start.py tapoi_interests &
#cd $ABSOLUTE_PATH../src && python cmp_start.py age_torch &
#cd $ABSOLUTE_PATH../src && python cmp_start.py gender_classifier
cd $ABSOLUTE_PATH../src && python cmp_start.py mimosa_clustering  &
cd $ABSOLUTE_PATH../src && python cmp_start.py sklearn_clustering_pca  &
cd $ABSOLUTE_PATH../src && python cmp_start.py mimosa_clustering_pca &
cd $ABSOLUTE_PATH../src && python cmp_start.py personas_generator --log D

