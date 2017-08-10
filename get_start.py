#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 21:27:56 2017

@author: sunguanxiong
"""

import tensorflow as tf

x = tf.placeholder(tf.float32);
w = tf.Variable([1.],tf.float32);
b = tf.Variable([0.],tf.float32);

y = w*x + b;

label = tf.placeholder(tf.float32);

loss = tf.reduce_sum(tf.square(label-y));

sess = tf.Session();
init = tf.global_variables_initializer();

#sess.run(init);

#print("The init loss is : ",sess.run(loss,{x:[1,2,3,4],label:[2,4,6,8]}));

optimizer = tf.train.GradientDescentOptimizer(0.01);
train = optimizer.minimize(loss);

sess.run(init);

for i in range(1000):
    print(i);
    sess.run(train,{x:[1,2,3,4],label:[2,4,6,8]})
    
w_curr, b_curr, loss_curr = sess.run([w,b,loss],{x:[1,2,3,4],label:[2,4,6,8]});
print("W: %s b: %s loss: %s"%(w_curr, b_curr, loss_curr));



