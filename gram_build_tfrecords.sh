#!/usr/bin/env bash
for DATASET in M-30 M-30-HD Urban1;
do
  DATASET_DIR=./GRAM-RTM/${DATASET}
  OUTPUT_DIR=${DATASET_DIR}
  python3 tf_convert_data.py \
      --dataset_name=gram \
      --dataset_dir=${DATASET_DIR} \
      --output_name=gram_${DATASET} \
      --output_dir=${OUTPUT_DIR}
done

