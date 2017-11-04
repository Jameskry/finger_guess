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
import os

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def video_to_frames(filename):
    for parent, dirnames, filenames in os.walk(filename):
        pass
    for i in range(0,len(filenames)):
        print(filename + filenames[i])
        vidcap = cv2.VideoCapture(filename + filenames[i])
        success,image = vidcap.read()
        count = 0
        video_frames_file_path = 'dataset/train_data' + '/' + filenames[i][:-4] + '/'
        if os.path.exists(video_frames_file_path):
            continue
        else:
            ensure_dir(video_frames_file_path)
        while 1:
            success,image = vidcap.read()
            if success:
                print('Read a new frame: frame%d.jpg' % count)
                cv2.imwrite(video_frames_file_path + 'frame%d.jpg' % count, image)     # save frame as JPEG file
                count += 1
            else:
                break

# video_to_frames(filename)

def create_dir():
    for i in range(0,70):
        count = int(i)
        if count < 10:
            count = '0' + str(count)
        else:
            count = str(count)
        video_frames_file_path = 'dataset/train_data/' + '100' + count
        if not os.path.exists(video_frames_file_path):
            os.makedirs(video_frames_file_path)

def rename_frame(rootDir):
    dirname = os.listdir(rootDir)
    def f(x):
        return rootDir + x + '/'
    dirname = list(map(f,dirname))

    for i in range(0,len(dirname)):
        count = 0
        files = os.listdir(dirname[i])
        files.sort(key=lambda x: int(x[5:-4]))
        for file in files:
            newname = str(count) + '.jpg'
            if os.path.isfile(os.path.join(dirname[i],newname)) == False:
                os.rename(os.path.join(dirname[i],file),os.path.join(dirname[i],newname))
            count += 1

def rename_dir():
    rootDir = 'dataset/train_data/'
    dirs = os.listdir(rootDir)
    # dirs.sort(key=lambda x: int(x[7:]))
    # for i in range(0,len(dirs)):
    #     count = int(i)
    #     if count < 10:
    #         count = '0' + str(count)
    #     else:
    #         count = str(count)
    #     newname = '000' + count
    #     os.rename(os.path.join(rootDir, dirs[i]), os.path.join(rootDir, newname))
    print(dirs)

# rename_frame( rootDir = 'dataset/test/')
rename_frame(rootDir = 'dataset/test_data/')