import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append('../')

from nets import ssd_vgg_512, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

from os import listdir, mkdir
from os.path import isfile, join
from PIL import Image
import argparse
import json


# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (512, 512)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_512.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
# ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
ckpt_filename = './checkpoints/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


def in_roi_check(roi, bbox):
  height, width = roi.shape
  bbox_ = [
    int(bbox[0] * width),
    int(bbox[1] * height),
    int(bbox[2] * width),
    int(bbox[3] * height),
  ]
  y_min = max(int(bbox_[1]) - 1, 0)
  y_max = min(int(bbox_[3]) - 1, height)
  x_min = max(int(bbox_[0]) - 1, 0)
  x_max = min(int(bbox_[2]) - 1, width)
  bbox_arr = np.zeros_like(roi)
  bbox_arr[y_min:y_max + 1, x_min:x_max + 1] = 1
  return (bbox_arr * roi).sum() > 0


def remove_bboxes_outside_roi(rclasses, rscores, rbboxes, roi):
  _rclasses = []
  _rscores = []
  _rbboxes = []
  for rclass, rscore, rbbox in zip(rclasses, rscores, rbboxes):
    if in_roi_check(roi, rbbox):
      _rclasses.append(rclass)
      _rscores.append(rscore)
      _rbboxes.append(rbbox)
  return np.asarray(_rclasses), np.asarray(_rscores), np.array(_rbboxes)


# Main image processing routine.
def process_image(img, roi, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
  # Run SSD network.
  rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                            feed_dict={img_input: img})

  # Get classes and bboxes from the net outputs.
  rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
    rpredictions, rlocalisations, ssd_anchors,
    select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

  rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
  rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
  rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
  # Resize bboxes to original image shape. Note: useless for Resize.WARP!
  rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
  rclasses, rscores, rbboxes = remove_bboxes_outside_roi(rclasses, rscores, rbboxes, roi)
  return rclasses, rscores, rbboxes


def read_roi(roi_path):
  roi = np.asarray(Image.open(roi_path), dtype=np.int32)
  roi = roi.sum(axis=2)
  roi.setflags(write=1)
  roi[roi <= 100] = 0
  roi[roi > 100] = 1
  return roi


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str)
  parser.add_argument('--roi_path', type=str)
  parser.add_argument('--select_threshold', type=float)
  parser.add_argument('--output_dir', type=str)
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = parse_args()
  get_files_in_dir = lambda dir_name: [f for f in listdir(dir_name) if isfile(join(dir_name, f))]

  no_of_files = len(get_files_in_dir(args.data_dir))
  roi = read_roi(args.roi_path)
  height, width = roi.shape

  try:
    mkdir(args.output_dir)
  except:
    pass
  try:
    mkdir(join(args.output_dir, 'visualization'))
  except:
    pass

  json.dump(vars(args), open(join(args.output_dir, 'args.txt'), 'wb'))
  print("Running. Please Wait...")
  with open(join(args.output_dir, 'bboxes.txt'), 'wb') as f:
    for img_index in range(no_of_files):
      img_path = join(args.data_dir, "image{:06d}.jpg".format(img_index + 1))
      img = mpimg.imread(img_path)
      rclasses, rscores, rbboxes =  process_image(img, roi)
      for rclass, rscore, rbbox in zip(rclasses, rscores, rbboxes):
        rbbox = [
          int(rbbox[0] * width),
          int(rbbox[1] * height),
          int(rbbox[2] * width),
          int(rbbox[3] * height),
        ]
        f.write('{}, -1, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, -1, -1, -1\n'.format(img_index + 1, rbbox[0], rbbox[1],
                                                                                      rbbox[2] - rbbox[0] + 1,
                                                                                      rbbox[3] - rbbox[1] + 1,
                                                                                      rscore))
      visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
      mpimg.imsave(join(args.output_dir, 'visualization', 'image{:06d}.png'.format(img_index + 1)), img)
  print("Finished!")
  # visualization.plt_bboxes(img, rclasses, rscores, rbboxes)