#!/usr/bin/env bash
for DATASET in M-30 M-30-HD Urban1;
do
  DATASET_DIR=GRAM-RTM/${DATASET}
  EVAL_DIR=./logs/${DATASET}
  CHECKPOINT_PATH=./checkpoints/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt
  CUDA_VISIBLE_DEVICES=0 python3 eval_ssd_network.py \
      --eval_dir=${EVAL_DIR} \
      --dataset_dir=${DATASET_DIR} \
      --dataset_name=gram \
      --dataset_split_name=${DATASET} \
      --model_name=ssd_512_vgg\
      --checkpoint_path=${CHECKPOINT_PATH} \
      --batch_size=32 | tee ~/ssd_eval_${DATASET}.log
done
