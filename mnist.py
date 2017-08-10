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

#NO RELU;NO regularize

w1 = tf.Variable(tf.truncated_normal(shape=[784,64],stddev=0.1));
b1 = tf.Variable(tf.constant(0.1,shape=[1,64]));
y1 = tf.matmul(x, w1) + b1;#[N,64]

w2 = tf.Variable(tf.truncated_normal(shape=[64,64],stddev=0.1));
b2 = tf.Variable(tf.constant(0.1,shape=[1,64]));
y_2 = tf.matmul(y1, w2) + b2;#[N,64]

w3 = tf.Variable(tf.truncated_normal(shape=[64,10],stddev=0.1));
b3 = tf.Variable(tf.constant(0.1,shape=[1,10]));
y_p = tf.matmul(y_2, w3) + b3;#[N,10]

#w = tf.Variable(tf.zeros(shape=[784,10]));
#b = tf.Variable(tf.zeros(shape=[1,10]));
#y_p = tf.matmul(x, w) + b;#[N,10]

cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_p))

optimizer = tf.train.GradientDescentOptimizer(0.1);
train_step = optimizer.minimize(cross_entropy);

#sess = tf.Session();
sess = tf.InteractiveSession();
sess.run(tf.global_variables_initializer());

for i in range(10000):
    
    minibatch = mnist.train.next_batch(100);
    loss, train = sess.run(
            [cross_entropy,train_step],{x : minibatch[0],y_ : minibatch[1]});
        
#    train_step.run({x:minibatch[0],y_:minibatch[1]});   
    print("Iter %s ----------> loss = %s "%(i,loss));
    
# Test trained model
correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("The training accuracy is : %s " %(sess.run(accuracy, 
                feed_dict={x: mnist.train.images,y_: mnist.train.labels})))

# Print ACCURACY ON TEST DATASET    
print("The testing accuracy is : %s " %(accuracy.eval(
                feed_dict={x:mnist.test.images,y_:mnist.test.labels})));