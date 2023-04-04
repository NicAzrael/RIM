from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

import imageio

import tensorflow as tf
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

from PIL import Image

def load_images(input_dir, batch_shape):
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in sorted(tf.io.gfile.glob(os.path.join(input_dir, '*.png')))[:1000]:
    with tf.io.gfile.GFile(filepath, "rb") as f:
      image = imageio.imread(f, pilmode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images

def save_images(images, filenames, output_dir):
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.io.gfile.GFile(os.path.join(output_dir, filename), 'w') as f:
      imageio.imsave(f, Image.fromarray(uint8((images[i, :, :, :] + 1.0) * 0.5 * 255)), format='png')

def graph(x, y, i, x_max, x_min, grad, grad2):
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  num_iter = FLAGS.num_iter
  batch_size = FLAGS.batch_size
  alpha = eps / num_iter
  momentum = FLAGS.momentum
  num_classes = 1001

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
     logits_v3, end_points_v3 = inception_v3.inception_v3(
        x, num_classes=num_classes, is_training=False)

  with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
     logits_v4, end_points_v4 = inception_v4.inception_v4(
         x, num_classes=num_classes, is_training=False)

  with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
     logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
         x, num_classes=num_classes, is_training=False)

  with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits_resnet, end_points_resnet = resnet_v2.resnet_v2_101(
        x, num_classes=num_classes, is_training=False)
     
  pred = tf.argmax(end_points_v3['Predictions'], 1)
  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)  
  logits = logits_v3
  cross_entropy = tf.compat.v1.losses.softmax_cross_entropy(one_hot,logits)
  auxlogits = end_points_res_v2['AuxLogits']
  cross_entropy += tf.compat.v1.losses.softmax_cross_entropy(one_hot,auxlogits,label_smoothing=0.0,weights=0.4)
  noise = tf.gradients(cross_entropy, x)[0]
  
  noise2 = grad2

#   resizing invariance in resnet
  x_2 = x
  with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits_resnet_2, end_points_resnet = resnet_v2.resnet_v2_101(
        input_diversity(x_2), num_classes=num_classes, is_training=False)
  cross_entropy_2 = tf.compat.v1.losses.softmax_cross_entropy(one_hot,logits_resnet_2)
  noise += tf.gradients(cross_entropy_2, x)[0]
 
  x_3 = x
  with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits_resnet_3, end_points_resnet = resnet_v2.resnet_v2_101(
        input_diversity(x_3), num_classes=num_classes, is_training=False)
  cross_entropy_3 = tf.compat.v1.losses.softmax_cross_entropy(one_hot,logits_resnet_3)
  noise += tf.gradients(cross_entropy_3, x)[0]

  x_4 = x
  with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits_resnet_4, end_points_resnet = resnet_v2.resnet_v2_101(
        input_diversity(x_4), num_classes=num_classes, is_training=False)
  cross_entropy_4 = tf.compat.v1.losses.softmax_cross_entropy(one_hot,logits_resnet_4)
  noise += tf.gradients(cross_entropy_4, x)[0]

  x_5 = x
  with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits_resnet_5, end_points_resnet = resnet_v2.resnet_v2_101(
        input_diversity(x_5), num_classes=num_classes, is_training=False)
  cross_entropy_5 = tf.compat.v1.losses.softmax_cross_entropy(one_hot,logits_resnet_5)
  noise += tf.gradients(cross_entropy_5, x)[0]
 
  #===============================MI-FGSM=======================================

  noise = noise / tf.reduce_mean(tf.abs(noise), [1,2,3], keep_dims=True)
  noise = momentum * grad + noise
  x = x + alpha * tf.sign(noise)

  #===============================I-FGSM=======================================

# =============================================================================
#   x = x + alpha * tf.sign(noise)
# =============================================================================
  
  x = tf.clip_by_value(x, x_min, x_max)
  i = tf.add(i, 1)
  return x, y, i, x_max, x_min, noise, noise2


def stop(x, y, i, x_max, x_min, grad, grad2):
  num_iter = FLAGS.num_iter
  return tf.less(i, num_iter)


#========================RIM=======================================================
def input_diversity(input_tensor):
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret = tf.cond(tf.random_uniform(shape=[1])[0] < 0.5, lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_height, FLAGS.image_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ret