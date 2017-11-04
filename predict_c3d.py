from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import c3d_model
import numpy as np
import matplotlib.pyplot as plt

NUM_CLASSES = 3
MAX_STEPS = 16
BATCH_SIZE = 10
LEARNING_RATE = 0.001
accuracy = []

def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
    labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
    return images_placeholder, labels_placeholder

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, wd):
    var = _variable_on_cpu(name, shape,
                           tf.contrib.layers.xavier_initializer())
    return var

def tower_acc(prediction, labels):
    correct_pred = tf.equal(tf.argmax(prediction, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def run_test():

    with tf.variable_scope('var_name') as var_scope:
        images_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)
        weights = {
            'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], wd=0.0005),
            'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], wd=0.0005),
            'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], wd=0.0005),
            'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], wd=0.0005),
            'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], wd=0.0005),
            'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], wd=0.0005),
            'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], wd=0.0005),
            'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], wd=0.0005),
            'wd1': _variable_with_weight_decay('wd1', [8192, 4096], wd=0.0005),
            'wd2': _variable_with_weight_decay('wd2', [4096, 4096], wd=0.0005),
            # 'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], wd=0.0005),
            'in': _variable_with_weight_decay('in', [4096, c3d_model.NUM_HIDDEN_UNIT], wd=0.0005),
            'out': _variable_with_weight_decay('out', [c3d_model.NUM_HIDDEN_UNIT, c3d_model.NUM_CLASSES], wd=0.0005)
        }
        biases = {
            'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
            'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
            'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
            'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
            'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
            'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
            'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
            'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
            'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
            'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
            # 'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.000),
            'in': _variable_with_weight_decay('in', [c3d_model.NUM_HIDDEN_UNIT], 0.000),
            'out': _variable_with_weight_decay('out', [c3d_model.NUM_CLASSES], 0.000)
        }

    dense1 = c3d_model.inference_c3d(images_placeholder, 1.0, BATCH_SIZE, weights, biases)
    logit = c3d_model.RNN(dense1, batch_size=BATCH_SIZE, weights=weights, biases=biases)
    prediction = tf.nn.softmax(logit)
    accuracy = tower_acc(prediction, labels_placeholder)

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    sess.run(init)
    saver.restore(sess, "my_net/save_net.ckpt")
    next_start_pos = 0
    acc_mean = 0.0
    acc_clip = []
    # print("weights:", sess.run(weights))
    # print("weights:", sess.run(biases))

    clip_start_pos = 0
    for step in xrange(MAX_STEPS):
        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        start_time = time.time()

        test_images, test_labels, _,_, _ = input_data.read_clip_and_label(
            filename='dataset/test_data/',
            batch_size=BATCH_SIZE,
            start_pos=next_start_pos,
            clip_start_pos= clip_start_pos,
            num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
            crop_size=c3d_model.CROP_SIZE,
            shuffle=True
        )
        if((step+1) % 2 == 0):
            clip_start_pos += 16
            next_start_pos = 0
        else:
            next_start_pos += 10
        print("test step: " + str(step+1))
        acc = sess.run(accuracy, feed_dict={images_placeholder: test_images,
                            labels_placeholder: test_labels
                                                })
        print("accuracy: " + "{:.5f}".format(acc))
        acc_mean += acc
        if((step+1) % 2 == 0):
            global acc_mean
            acc_mean = acc_mean/2
            acc_clip.append(acc_mean)
            acc_mean = 0.0

    return acc_clip

accuracy = run_test()
# accuracy = [0.05000000074505806, 0.20000000298023224, 0.35000002384185791, 0.35000002384185791, 0.40000000596046448, 0.35000002384185791, 0.15000000596046448, 0.20000000298023224]
print(accuracy)
plt.plot(accuracy)
plt.xlabel('clip')
plt.ylabel('accuracy')
plt.title('16 frames per clip, equal to 66.67ms')
plt.show()
