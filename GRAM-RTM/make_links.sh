#!/usr/bin/env bash
GRAM_DIR=/home/nhat/Engage_Grant_Project/Data/GRAM-RTM/GRAM-RTMv4
for DATASET in M-30 M-30-HD Urban1;
do
  rm ${DATASET} -rf
  mkdir ${DATASET}
  ln -s ${GRAM_DIR}/Annotations/${DATASET}/xml ${DATASET}/Annotations
  ln -s ${GRAM_DIR}/Images/${DATASET}/ ${DATASET}/JPEGImages
done