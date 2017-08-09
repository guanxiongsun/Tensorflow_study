#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 22:57:37 2017

@author: sunguanxiong
"""

import tensorflow as tf
#import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None,784]);
y_ = tf.placeholder(tf.float32, shape=[None,10]);

w = tf.Variable(tf.zeros(shape=[784,10]));
b = tf.Variable(tf.zeros(shape=[1,10]));

y_p = tf.matmul(x, w)+b;

cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_p))

optimizer = tf.train.GradientDescentOptimizer(0.1);
train_step = optimizer.minimize(cross_entropy);

#sess = tf.Session();
sess = tf.InteractiveSession();
sess.run(tf.global_variables_initializer());

for i in range(1000):
    
    minibatch = mnist.train.next_batch(100);
    loss, train = sess.run(
            [cross_entropy,train_step],{x : minibatch[0],y_ : minibatch[1]});
        
#    train_step.run({x:minibatch[0],y_:minibatch[1]});   
    print("Iter %s ----------> loss = %s "%(i,loss));
    
# Test trained model
correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("The training accuracy is : %s " %(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels})))