#!/bin/bash
set -e

# model="log/2016-06-08-13-28-05.652-dstc-original-split-OrD0.7w100e20/2016-06-08-13-28-05.652dstc-original-split-OrD0.7w100e20-reward-0.6917-step-0004528"
model="$1"; shift
# validate_to_dir="log/TEST-${name}"
validate_to_dir="$1"; shift
# pickled_dataset_with_train=dstc2_original_dst_prefix_TEST
pickled_dataset_with_train="$1"; shift
# dstc2_set_name=dstc2_test
dstc2_set_name=$1; shift

dirname="$(dirname $model)"
json_config="$(ls $dirname/*.json)"
name="`basename $json_config`"; 
val_out="$validate_to_dir/validation.out"

./demo.py  \
  --validate_output "$val_out" \
  --load_dstc_dir  "$pickled_dataset_with_train" \
  --validate_to_dir "$validate_to_dir" \
  --config "$json_config" \
  --load_model "$model"

printf "Validation stored to $validate_to_dir\n"

printf "Backing up also models - todo backup vocabs too?\n"
cp $model $json_config $validate_to_dir

printf "Converting validation output to DSTC2 format\n"
dst_valout="${val_out}_dstc.json"
./data/dstc2_original_split/validate2dstc.py "$val_out" ./data/dstc2_original_split/tmp/scripts/config/${dstc2_set_name}.flist "$dst_valout"

printf "Running DSTC2 evaluation scrpts\n"
./data/dstc2_original_split/tracker_scorer.sh ./data/dstc2_original_split/tmp "$dst_valout" $dstc2_set_name
cat "${dst_valout}.txt"
