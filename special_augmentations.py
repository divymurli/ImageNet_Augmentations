from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os, sys
import numpy as np
from imagenet_c import corrupt
import PIL.Image
import IPython.display as ip_disp
from cStringIO import StringIO

def random_rotation(image):
    image_rotated = tf.image.rot90(image, k=np.random.randint(0,4))
    return image_rotated

#Using imagenet_c corrupt library, but this is slow when run on GPU
def tf_corrupt(image, severity, corruption_name):
    out = tf.py_func(corrupt, [image, severity, corruption_name], tf.uint8)
    out.set_shape(image.shape)
    return out

#TensorFlow version of the Gaussian Noise function implemented in NumPy in the imagenet_c package
def gaussian_noise(image, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    image = tf.cast(image, tf.float32)
    image = image/255.
    noise = tf.random_normal(shape=tf.shape(image), mean = 0.0, stddev = c, dtype = tf.float32)
    out = tf.clip_by_value(image + noise, 0, 1)*255
    out = tf.cast(out, tf.uint8)
    return out

#Standard Gaussian kernel where size must be specified, along with Gaussian parameters
def gaussian_kernel(size, mean_x, mean_y, std_x, std_y):
    d_x = tf.distributions.Normal(mean_x, std_x)
    d_y = tf.distributions.Normal(mean_y, std_y)
    vals_x = d_x.prob(tf.range(start = -size, limit = size+1, dtype = tf.float32))
    vals_y = d_y.prob(tf.range(start = -size, limit = size+1, dtype = tf.float32))
    
    gauss_kernel = tf.einsum('i,j->ij', vals_y, vals_x)

    return gauss_kernel/tf.reduce_sum(gauss_kernel)

#Gaussian kernel implemented as in skimage.filters.gaussian, for TensorFlow. Sourcecode available here:
#http://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian
def gaussian_kernel2(mean, std): #set truncate parameter to 4, hardcoded for now
    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start = -int(4*std + 0.5), limit = int(4*std+0.5) + 2, dtype = tf.float32 ))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)

    return gauss_kernel/tf.reduce_sum(gauss_kernel)

def gaussian_blur(img_batch, severity=1):
    c = [1., 2., 3., 4., 6.][severity - 1]
    img_batch = tf.cast(img_batch, tf.float32)
    img_batch = img_batch/255.
    #image = tf.expand_dims(image, 0)
    gauss_filter = gaussian_kernel2(0.0, c)
    gf = tf.expand_dims(gauss_filter, 2)
    gf = tf.expand_dims(gf, 3)
    
   #convolutions
    blurred_channel_1 = tf.nn.conv2d(tf.expand_dims(img_batch[:, :, :, 0], 3), gf, strides=[1, 1, 1, 1], padding="SAME")
    blurred_channel_2 = tf.nn.conv2d(tf.expand_dims(img_batch[:, :, :, 1], 3), gf, strides=[1, 1, 1, 1], padding="SAME")
    blurred_channel_3 = tf.nn.conv2d(tf.expand_dims(img_batch[:, :, :, 2], 3), gf, strides=[1, 1, 1, 1], padding="SAME")
    
    stacked = tf.concat([blurred_channel_1, blurred_channel_2, blurred_channel_3], 3)
    out = tf.clip_by_value(stacked, 0, 1)*255

    return out

def blur_except_block(img_batch, min_block_size, max_block_size, severity=1):
    def _get_offset_and_len(crop_size):
        random_len = tf.random_uniform(
                [], minval=min_block_size, 
                maxval=max_block_size, 
                dtype=tf.int32
                )
        random_offset = tf.random_uniform(
                [], minval=0,
                maxval=1, dtype=tf.float32,
                )
        random_offset = tf.cast(
                random_offset * tf.cast((crop_size-random_len), tf.float32),
                tf.int32)
        return random_offset, random_len
    
    #store blurred image
    img_batch_blur = gaussian_blur(img_batch, severity)

    x_offset, x_len = _get_offset_and_len(img_batch.get_shape().as_list()[1])
    y_offset, y_len = _get_offset_and_len(img_batch.get_shape().as_list()[2])

    x_indx = tf.range(x_offset, x_offset + x_len)
    x_indx = tf.tile(tf.expand_dims(x_indx, axis=1), [1, y_len])

    y_indx = tf.range(y_offset, y_offset + y_len)
    y_indx = tf.tile(tf.expand_dims(y_indx, axis=0), [x_len, 1])
    
    x_indx = tf.reshape(x_indx, [-1,1])
    y_indx = tf.reshape(y_indx, [-1,1])
    
    batch_indx = tf.range(0,img_batch.get_shape()[0])
    batch_indx = tf.tile(tf.expand_dims(batch_indx, axis=1),[1,x_len*y_len])
    batch_indx = tf.reshape(batch_indx, [-1,1])

    all_indx = tf.concat(
            [batch_indx, tf.tile(x_indx, [img_batch.get_shape()[0],1]), tf.tile(y_indx, [img_batch.get_shape()[0],1])],
            axis=-1)
    blur_pixel = tf.slice(img_batch_blur, [0, x_offset, y_offset, 0], [img_batch.get_shape().as_list()[0], x_len, y_len, 3])
    clean_pixel = tf.slice(img_batch, [0, x_offset, y_offset, 0], [img_batch.get_shape().as_list()[0], x_len, y_len, 3])

    blur_pixel = tf.reshape(blur_pixel, [-1, 3])
    clean_pixel = tf.reshape(clean_pixel,[-1, 3])

    blur_pixel = tf.cast(blur_pixel, tf.int32)
    clean_pixel = tf.cast(clean_pixel, tf.int32)

    clean_pixel = tf.scatter_nd(all_indx, clean_pixel, img_batch.get_shape().as_list())

    add_pixel = -blur_pixel
    add_pixel = tf.scatter_nd(all_indx, add_pixel, img_batch.get_shape().as_list())

    img = tf.cast(tf.cast(img_batch_blur, tf.int32) + add_pixel + clean_pixel, tf.uint8)

    return img


