# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time

def get_frames_data(frame_dir, num_frames_per_clip=16, start_pos = 0, shuffle=None):
    ret_arr = []
    s_index = start_pos
    e_index = s_index + num_frames_per_clip
    framenames = []
    for _, _, framenames in os.walk(frame_dir):
        if(len(framenames)<num_frames_per_clip):
            return [], s_index
    # filenames = sorted(filenames)
    framenames.sort(key=lambda x: int(x[:-4]))
    if (shuffle):
        s_index = random.randint(0, len(framenames) - num_frames_per_clip)   # start index
        e_index = s_index + num_frames_per_clip
    for i in range(s_index, e_index):
        image_name =frame_dir + str(i) + '.jpg'
        img = Image.open(image_name)
        img_data = np.array(img)
        ret_arr.append(img_data)
    return ret_arr, e_index


def read_clip_and_label(filename, batch_size, start_pos=0, clip_start_pos = 0,
                        num_frames_per_clip=16, crop_size=112, shuffle = None):
    batch = []
    label = []
    dirnames = []
    after_shuffle_dirname = []
    img_datas = []
    tmp_label = 0
    batch_index = 0
    next_batch_start = start_pos
    isShuffle = shuffle

    for _, dirnames, _ in os.walk(filename):
        # print(dirnames)
        break
    frames_all_filenames = dirnames

    if shuffle:
        video_indices = list(range(len(frames_all_filenames)))
        random.seed(time.time())
        random.shuffle(video_indices)
    else:
        # Process videos sequentially
        video_indices = range(start_pos, len(frames_all_filenames))

    for index in video_indices:
        if (batch_index >= batch_size):
            next_batch_start = index
            break
        dirname = frames_all_filenames[index]
        after_shuffle_dirname.append(dirname)
        dirname_abs = filename + dirname + '/'
        # global tmp_label
        tmp_label = int(dirname[0])
        if not shuffle:
            print("Loading a video clip from {}...".format(dirname))
            print('Load video file name :',dirname_abs)
        tmp_data, clip_end_index = get_frames_data(dirname_abs, num_frames_per_clip, start_pos=clip_start_pos, shuffle=isShuffle)  # 16帧图像
        # global img_datas
        img_datas = []
        if (len(tmp_data) != 0):
            # crop image then add to list[img_datas]
            for j in xrange(len(tmp_data)):
                img = Image.fromarray(tmp_data[j].astype(np.uint8))  # array -> image
                if (img.width > img.height):
                    scale = float(crop_size) / float(img.height)
                    img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1),
                                                              crop_size))).astype(np.float32)
                else:
                    scale = float(crop_size) / float(img.width)
                    img = np.array(cv2.resize(np.array(img), (crop_size,
                                                              int(img.height * scale + 1)))).astype(np.float32)
                crop_x = int((img.shape[0] - crop_size) / 2)
                crop_y = int((img.shape[1] - crop_size) / 2)
                img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
                img_datas.append(img)
            batch.append(img_datas)
            label.append(int(tmp_label))
            batch_index = batch_index + 1

    # print(after_shuffle_dirname)

    # If number of remain videos < BATCH_SIZE
    # Duplicate the last video into batch for batch_size - valid_len times
    valid_len = len(batch)
    pad_len = batch_size - valid_len
    if pad_len:
        for i in range(pad_len):
            batch.append(img_datas)
            label.append(int(tmp_label))

    np_arr_data = np.array(batch).astype(np.float32)
    np_arr_label = np.array(label).astype(np.int64)

    return np_arr_data, np_arr_label, next_batch_start, dirnames, valid_len