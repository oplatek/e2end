#!/bin/bash
model="log/2016-06-08-13-28-05.652-dstc-original-split-OrD0.7w100e20/2016-06-08-13-28-05.652dstc-original-split-OrD0.7w100e20-reward-0.6917-step-0004528"
validate_to_dir="log/TEST-${name}"

dirname="$(dirname $model)"
json_config="$(ls $dirname/*.json)"
name=`basename $json_config`; 

rm $validate_to_dir/*/*
rmdir $validate_to_dir/*
rm $validate_to_dir/*
rmdir $validate_to_dir

set -e
./demo.py  \
  --load_dstc_dir dstc2_original_dst_prefix_TEST \
  --validate_to_dir $validate_to_dir \
  --config $json_config \
  --load_model $model

echo validation stored to $validate_to_dir

echo "Backing up also models - todo backup vocabs too?"
cp $model $json_config $validate_to_dir
