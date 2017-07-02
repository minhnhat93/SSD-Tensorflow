#!/usr/bin/env bash
mkdir bboxes
for SELECT_THRESHOLD in 0.0 0.1 0.25 0.5;
do
  for DATASET in M-30 M-30-HD Urban1;
  do
    DATA_DIR=GRAM-RTM/${DATASET}/JPEGImages
    ROI_PATH=GRAM-RTM/${DATASET}/roi_map.jpg
    OUTPUT_DIR=bboxes/${DATASET}_${SELECT_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python gram_create_ssd_bboxes.py \
     --data_dir ${DATA_DIR} \
     --roi_path ${ROI_PATH} \
     --select_threshold ${SELECT_THRESHOLD} \
     --output_dir ${OUTPUT_DIR}
  done
done
