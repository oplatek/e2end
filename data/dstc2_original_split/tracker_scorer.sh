#!/bin/bash
#Usage: ./tracker_scorer.sh results.json dstc2_test
set -e

DSTC_HOME=$1 ; shift
LOGFILE=$1; shift
DSTC_DATASET=$1; shift
SCOREFILE=$LOGFILE.csv
REPORTFILE=$LOGFILE.txt
# python $DSTC_HOME/scripts/score.py  --rocbins=2 --dataset $DSTC_DATASET --dataroot $DSTC_HOME/data --trackfile $LOGFILE --scorefile $SCOREFILE --ontology $DSTC_HOME/scripts/config/ontology_dstc2.json
python $DSTC_HOME/scripts/score.py  --dataset $DSTC_DATASET --dataroot $DSTC_HOME/data --trackfile $LOGFILE --scorefile $SCOREFILE --ontology $DSTC_HOME/scripts/config/ontology_dstc2.json

echo score script done
python $DSTC_HOME/scripts/report.py --scorefile $SCOREFILE  > $REPORTFILE
echo report script done
