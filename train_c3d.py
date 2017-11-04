import os
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import c3d_model
import math
import numpy as np
import matplotlib.pyplot as plt


NUM_CLASSES = 3
MAX_STEPS = 5000
BATCH_SIZE = 10
LEARNING_RATE = 0.001
loss_per_step = []


def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
    labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
    return images_placeholder, labels_placeholder


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.contrib.layers.xavier_initializer())
  # if wd:
  #   weight_decay = tf.nn.l2_loss(var) * wd
  #   tf.add_to_collection('losses', weight_decay)
  return var


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference_c3d().
    labels: Labels from inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # One-hot labels
  # labels[BATCH_SIZE,]

  # sparse_labels = tf.reshape(labels, [BATCH_SIZE, 1])
  # indices = tf.reshape(tf.range(BATCH_SIZE), [BATCH_SIZE, 1])
  # concated = tf.concat([tf.to_int64(indices), sparse_labels], 1)
  # dense_labels = tf.sparse_to_dense(concated,
  #                                   [BATCH_SIZE, NUM_CLASSES],
  #                                   1.0, 0.0)
  one_hot = tf.one_hot(labels, 3, 1.0, 0.0, axis=-1)

  # Calculate the average cross entropy loss across the batch.
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=one_hot, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss'),one_hot
  # return cross_entropy_mean,dense_labels


def tower_acc(prediction, labels):
    correct_pred = tf.equal(tf.argmax(prediction, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def run_training():
  # Get the sets of images and labels for training, validation, and
  # Tell TensorFlow that the model will be built into the default Graph.

  # Create model directory
    global loss_per_step
    with tf.Graph().as_default():
        # global_step = tf.get_variable(
        #     'global_step',
        #     [],
        #     initializer=tf.constant_initializer(0),
        #     trainable=False
        # )
        images_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)
        with tf.variable_scope('var_name') as var_scope:
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

        dense1 = c3d_model.inference_c3d(images_placeholder, 0.6,BATCH_SIZE, weights, biases)
        logit = c3d_model.RNN(dense1, batch_size=BATCH_SIZE, weights=weights, biases=biases)

        total_loss,one_hot_labels = loss(logit, labels_placeholder)
        prediction = tf.nn.softmax(logit)
        accuracy = tower_acc(prediction, labels_placeholder)
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(total_loss)
        # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(total_loss)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(init)

        saver = tf.train.Saver()
        save_path = saver.save(sess, "my_net/save_net.ckpt")
        print("Save to path: ", save_path)

        # sess.run(tf.Print(weights['wc1'],[weights['wc1']],message='wc1:',summarize=100))


    plt.ion()
    for step in xrange(MAX_STEPS):
        start_time = time.time()
        train_images, train_labels, _, _, _ = input_data.read_clip_and_label(
            filename='dataset/train_data/',
            batch_size=BATCH_SIZE,
            num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
            crop_size=c3d_model.CROP_SIZE,
            shuffle=True
        )
        sess.run(train_step, feed_dict={
            images_placeholder: train_images,
            labels_placeholder: train_labels
        })
        duration = time.time() - start_time
        print('Step %d: %.3f sec' % (step, duration))

        step_loss = sess.run(total_loss, feed_dict={images_placeholder: train_images,
                                                    labels_placeholder: train_labels
                                                    })
        print("loss: " + "{:.5f}".format(step_loss))
        loss_per_step.append(step_loss)

        # plt.plot(loss_per_step)
        # plt.xlabel('clip')
        # plt.ylabel('loss')
        # plt.title('16 frames per clip, equal to 66.67ms')
        # plt.pause(0.05)
        # while True:
        #     plt.pause(0.05)


        # sess.run(tf.Print(images_placeholder, [images_placeholder], message='images_placeholder:', summarize=100),
        #          feed_dict={images_placeholder: train_images})
        # sess.run(tf.Print(labels_placeholder, [labels_placeholder], message='labels:', summarize=100),
        #          feed_dict={labels_placeholder: train_labels})
        # sess.run(tf.Print(one_hot_labels, [labels_placeholder], message='one_hot_labels:', summarize=100),
        #          feed_dict={labels_placeholder: train_labels})
        # sess.run(tf.Print(conv1, [conv1], message='conv1:', summarize=100),
        #          feed_dict={images_placeholder: train_images})
        # sess.run(tf.Print(tf.shape(conv1), [tf.shape(conv1)], message='conv1.shape:', summarize=100),
        #          feed_dict={images_placeholder: train_images})
        # sess.run(tf.Print(pool1, [pool1], message='pool1:', summarize=100),
        #          feed_dict={images_placeholder: train_images})
        # sess.run(tf.Print(tf.shape(pool1), [tf.shape(pool1)], message='pool1.shape:', summarize=100),
        #          feed_dict={images_placeholder: train_images})
        # sess.run(tf.Print(dense1, [dense1], message='dense1:', summarize=100),
        #          feed_dict={images_placeholder: train_images})
        # sess.run(tf.Print(dense2, [dense2], message='dense2:', summarize=100),
        #          feed_dict={images_placeholder: train_images})
        # sess.run(tf.Print(logit, [logit], message='Logit:', summarize=100),
        #          feed_dict={images_placeholder: train_images})

        if (step) % 20 == 0:
            print('Training Data Eval:')
            test_images, test_labels, _, _, _ = input_data.read_clip_and_label(
                filename='dataset/test_data/',
                batch_size=BATCH_SIZE,
                num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
                crop_size=c3d_model.CROP_SIZE,
                shuffle= True
            )
            acc = sess.run(accuracy, feed_dict={images_placeholder: test_images,
                            labels_placeholder: test_labels
                                                })
            print("accuracy: " + "{:.5f}".format(acc))

    return loss_per_step

def main(_):
  run_training()

loss_list = run_training()
plt.plot(loss_list)
plt.xlabel('clip')
plt.ylabel('loss')
plt.title('16 frames per clip, equal to 66.67ms')
plt.show()
