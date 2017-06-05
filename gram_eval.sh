#!/usr/bin/env bash
for DATASET in M-30 M-30-HD Urban1;
do
  DATASET_DIR=GRAM-RTM/M-30
  EVAL_DIR=./logs/${DATASET}
  CHECKPOINT_PATH=./checkpoints/ssd_300_vgg.ckpt
  python3 eval_ssd_network.py \
      --eval_dir=${EVAL_DIR} \
      --dataset_dir=${DATASET_DIR} \
      --dataset_name=gram \
      --dataset_split_name=M-30 \
      --model_name=ssd_300_vgg \
      --checkpoint_path=${CHECKPOINT_PATH} \
      --batch_size=1 | tee ~/ssd_eval_${DATASET}.log
done
