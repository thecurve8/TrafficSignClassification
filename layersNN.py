# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:10:44 2020

@author: Alexander

This script defines some layers that will be used in the NN network

This scrip requires tensorflow
"""
import tensorflow as tf

def conv2d(x, W, b, strides=1):
    """Wrapper for a 2D convolutional layer with bias and activation function ReLu
    
    Padding is valid so the output tensor's size decreases horizontally and vertically by the size of the kernel-1
    
    Parameters
    ----------
    x : tensor
        the input tensor for this layer
    W : tensor
        the weight tensor for this layer
    b : tensor
        the bias tensor for this layer
    strides : int, optional
        Horizontal and vertical Strides to be used in the convolution (default is 1)
        
    Returns
    -------
    tensor reprsenting the output of the layer
    """ 
    
    conv = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    convBias = tf.nn.bias_add(conv, b)
    return tf.nn.relu(convBias)

def max_pooling2d(x, pool_size=2):
    """Wrapper for a max pooling layer
    
    The size of the output tensor is the one of the input tensor divided by pool_size
    
    Parameters
    ----------
    x : tensor
        the input tensor for this layer
    pool_size : int, optional
        Size of the square region used for max_pool (default is 2)
        
    Returns
    -------
    tensor reprsenting the output of the layer
    """ 
    
    return tf.nn.max_pool(x, ksize=[1,pool_size,pool_size,1], strides=[1,pool_size,pool_size,1], padding="VALID", data_format="NHWC")

def dropout(x, keep_prob):
    """Wrapper for a dropout layer
        
    Parameters
    ----------
    x : tensor
        the input tensor for this layer
    keep_prob : float
        probaility that a neuron of the previous layer keeps its value
        
    Returns
    -------
    tensor reprsenting the output of the layer with some nurons set to 0 according to keep_prob
    """ 
    
    return tf.nn.dropout(x,keep_prob)

def flatten(x, height, width, channels):
    """Wrapper for a flatten layer
        
    Parameters
    ----------
    x : tensor
        the input tensor for this layer
    height : int
        height of the input tensor
    width : int
        width of the input tensor
    channels : int
        channels of the input tensor
        
    Returns
    -------
    tensor reprsenting the output of the layer, flattened version of the input tensor
    """     
    
    return tf.reshape(x, [-1, height*width*channels])

def dense(x, W, b, activation='none'):
    """Wrapper for a dense layer, with no activation or ReLu activation
       
    Parameters
    ----------
    x : tensor
        the input tensor for this layer
    W : tensor
        the weight tensor for this layer
    b : tensor
        the bias tensor for this layer
    activation : str, optional
        activation function, only avaiable is "relu" (default is "none")
        
    Returns
    -------
    out
        tensor reprsenting the output of the layer
    """    
    
    out = tf.add(tf.matmul(x, W), b)
    if activation=='relu':
        out = tf.nn.relu(d)
    return out