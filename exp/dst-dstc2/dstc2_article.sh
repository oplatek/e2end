#!/bin/bash
repo_root=$(git rev-parse --show-toplevel)
cd $repo_root

dropout=0.7
wemb=100
encoder_size=100
name="drop${dropout}-wemb${wemb}-enc_size${encoder_size}"

echo "qsubmit is wrapper around qsub which is not part of the repository"

qsubmit --mem 12g --jobname $name \
  "source /home/oplatek/virtualenv/tensorflow-cpu/bin/activate; \
  ./demo.py --cluster --history_prefix --dst --num_buckets 10 \
  --train_file ./data/dstc2_original_split/data.dstc2.train.json \
  --dev_file ./data/dstc2_original_split/data.dstc2.dev.json \
  --word_embed_size ${wemb} --encoder_size ${encoder_size} --enc_dropout_keep $dropout \
  --exp encdec-dstc-original-split-$name; \
  "

# qsubmit --mem 12g --jobname $name \
#   "source /home/oplatek/virtualenv/tensorflow-cpu/bin/activate; \
#   ./demo.py --cluster --history_prefix --dst --num_buckets 10 \
#   --train_file ./data/dstc2/data.dstc2.train.json \
#   --dev_file ./data/dstc2/data.dstc2.dev.json \
#   --word_embed_size ${wemb} --encoder_size ${encoder_size} --enc_dropout_keep $dropout \
#   --exp encdec-dstc-new-split-$name; \
#   "
#
# qsubmit --mem 12g --jobname $name \
#   "source /home/oplatek/virtualenv/tensorflow-cpu/bin/activate; \
#   ./demo.py --cluster --history_prefix --dst --num_buckets 10 \
#   --model DstIndep \
#   --train_file ./data/dstc2_original_split/data.dstc2.train.json \
#   --dev_file ./data/dstc2_original_split/data.dstc2.dev.json \
#   --word_embed_size ${wemb} --encoder_size ${encoder_size} --enc_dropout_keep $dropout \
#   --exp indep-dstc-original-split-$name; \
#   "
#
# qsubmit --mem 12g --jobname $name \
#   "source /home/oplatek/virtualenv/tensorflow-cpu/bin/activate; \
#   ./demo.py --cluster --history_prefix --dst --num_buckets 10 \
#   --model DstIndep \
#   --train_file ./data/dstc2/data.dstc2.train.json \
#   --dev_file ./data/dstc2/data.dstc2.dev.json \
#   --word_embed_size ${wemb} --encoder_size ${encoder_size} --enc_dropout_keep $dropout \
#   --exp indep-dstc-new-split-$name; \
#   "
