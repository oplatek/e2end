#!/bin/bash
repo_root=$(git rev-parse --show-toplevel)
cd $repo_root

for dropout in 0.6 0.65 0.7 1.0 ; do
  for wemb in 10 100 150 200 300 ; do 
    for encoder_size in 20 100 200 300 400 600 ; do
      name=OrD${dropout}w${wemb}e${encoder_size}
      qsubmit --mem 12g --jobname $name \
        "source /home/oplatek/virtualenv/tensorflow-cpu/bin/activate; \
        ./demo.py --cluster --history_prefix --dst --load_dstc_dir dstc2_original_dst_prefix --num_buckets 10 \
        --train_file ./data/dstc2_original_split/data.dstc2.train.json \
        --dev_file ./data/dstc2_original_split/data.dstc2.dev.json \
        --word_embed_size ${wemb} --encoder_size ${encoder_size} --enc_dropout_keep $dropout \
        --exp dstc-original-split-$name"
    done
  done
done
