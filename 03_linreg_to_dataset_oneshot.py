"""
This is a simple imitation of its counterpart
"""

import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import os
import util
import time
import math

"""
The model is like this:
inference: Find a relationship between X and Y to predict Y from X
Inference: Y_predicted = w * X + b
MSE = E[(y - y_predicted)^2]
"""

cwd = os.getcwd()
FILE_PATH = cwd + '/data/birth_life_2010.txt'
_, data, len_data, _ = util.read_birth_file(FILE_PATH)

epoch_num = 160
dataset = tf.contrib.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))
dataset = dataset.repeat(epoch_num)
pass

# There are two ways to iterate through different epochs
#1 
iterator = dataset.make_one_shot_iterator()
X, Y = iterator.get_next()
# this way we iterates through the dataset just once, so no initialization
# so we have to assign X, Y every time we go into the next running of the session
# by get_next we realize this function

# create weight and bias, initialized to 0
w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

Y_pred = X * w + b

# sqrloss = tf.square(Y - Y_pred, name='loss')
sqrloss = util.huber_loss(Y, Y_pred)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(sqrloss)

tic = time.time()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) 
    writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)

    # The iterator has to initalized anytime we start a epoch
    total_loss = 0
    iter = 0
    try:
        while True:
            _, sqrloss_ = sess.run([optimizer, sqrloss])
            total_loss += sqrloss_
            if iter%len_data == 0:
                print('Epoch {0}: {1}'.format(int(math.floor(iter/len_data)), total_loss/len_data))
                total_loss = 0
            iter += 1

    except tf.errors.OutOfRangeError:
        pass
        # when we are running out of all dataset entries, tf will raise OutOfRangeError
        # we have to catch the exception by outselves.
    
    w_out, b_out = sess.run([w, b]) 
    print('w: %f, b: %f' %(w_out, b_out))



writer.close()
toc = time.time()
print('Took: %f seconds' %(toc- tic))

# plot the results
plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data with squared error')
# plt.plot(data[:,0], data[:,0] * (-5.883589) + 85.124306, 'g', label='Predicted data with Huber loss')
plt.legend()
plt.show()