#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:56:34 2017

@author: sunguanxiong
"""

import tensorflow as tf
#import numpy as np

def weight_varibles(shape):
    init = tf.truncated_normal(shape,stddev=0.1);
    return tf.Variable(init);

def bias_varibles(shape):
    init = tf.constant(0.1,shape=shape);
    return tf.Variable(init);

#input: A Tensor. Must be one of the following types: half, float32. 
#        A 4-D tensor. The dimension order is interpreted according 
#        to the value of data_format, see below for details.
#        
#filter: A Tensor. Must have the same type as input. A 4-D tensor of
#        shape [filter_height, filter_width, in_channels, out_channels]
#        
#strides: A list of ints. 1-D tensor of length 4. The stride of the 
#        sliding window for each dimension of input. The dimension 
#        order is determined by the value of data_format, see below for details.
#        
#padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
#
#use_cudnn_on_gpu: An optional bool. Defaults to True.
#
#data_format: An optional string from: "NHWC", "NCHW". Defaults to "NHWC". 
#        Specify the data format of the input and output data. With the 
#        default format "NHWC", the data is stored in the order of: 
#        [batch, height, width, channels]. Alternatively, the format 
#        could be "NCHW", the data storage order of: 
#            [batch, channels, height, width].
def conv2d(x,W):
    return tf.nn.relu(tf.nn.conv2d(x,W,[1,1,1,1],'SAME',True));

def max_pooling_2x2(x):
    return tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],'SAME')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None,784]);
y_ = tf.placeholder(tf.float32, shape=[None,10]);
keep_prob = tf.placeholder(tf.float32);

x_image = tf.reshape(x,[-1,28,28,1]);

W_conv1 = weight_varibles([5,5,1,32]);
b_conv1 = bias_varibles([32])

h_conv1 = conv2d(x_image,W_conv1);
h_pool1 = max_pooling_2x2(h_conv1);#[N,14,14,1]

W_conv2 = weight_varibles([5,5,32,64]);
b_conv2 = bias_varibles([64]);

h_conv2 = conv2d(h_pool1, W_conv2);
h_pool2 = max_pooling_2x2(h_conv2);#[N,7,7,1]

flaten = tf.reshape(h_pool2,[-1,7*7*64]);
W_fc1 = weight_varibles([7*7*64,128]);
b_fc1 = bias_varibles([128]);

h_fc1 = tf.nn.relu(tf.matmul(flaten, W_fc1) + b_fc1);#[N,128];

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob);
W_fc2 = weight_varibles([128,128]);
b_fc2 = bias_varibles([128]);

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2);

h_f2_drop = tf.nn.dropout(h_fc2, keep_prob);
W_fc3 = weight_varibles([128, 10]);
b_fc3 = bias_varibles([10]);

y_p = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3);

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_p);

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy);

correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(y_, 1));
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(5000):
    batch = mnist.train.next_batch(256)
    if i % 10 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob:1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5});
    if i % 500 == 0:
      test_accuracy = accuracy.eval(
              {x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0 });
      print('step %d, testing accuracy ----------> %g' % (i, test_accuracy));


