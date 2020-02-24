# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:10:44 2020

@author: Alexander
"""
import tensorflow as tf

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    conv = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    convBias = tf.nn.bias_add(conv, b)
    return tf.nn.relu(convBias)

def max_pooling2d(x, pool_size=2):
    return tf.nn.max_pool(x, ksize=[1,pool_size,pool_size,1], strides=[1,pool_size,pool_size,1], padding="VALID", data_format="NHWC")

def dropout(x, keep_prob):
    return tf.nn.dropout(x,keep_prob)

def flatten(x, height, width, channels):
    return tf.reshape(x, [-1, height*width*channels])

def dense(x, W, b, activation='none'):
    d = tf.add(tf.matmul(x, W), b)
    if activation=='relu':
        d = tf.nn.relu(d)
    return d