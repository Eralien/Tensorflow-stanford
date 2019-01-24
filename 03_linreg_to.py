"""
This is a simple imitation of its counterpart
"""

import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import os
import util
import time

"""
The model is like this:
inference: Find a relationship between X and Y to predict Y from X
Inference: Y_predicted = w * X + b
MSE = E[(y - y_predicted)^2]
"""

cwd = os.getcwd()
FILE_PATH = cwd + '/data/birth_life_2010.txt'
_, data, len_data, _ = util.read_birth_file(FILE_PATH)
pass

# Create placeholders X and Y
X = tf.placeholder(tf.float32, name='Birth_Rate')
Y = tf.placeholder(tf.float32, name='Life_Expectancy')

# Create weight and bias with initial value both 0 
w = tf.get_variable("weight", initializer=tf.constant(0.0))
b = tf.get_variable("bias", initializer=tf.constant(0.0))

Y_pred = w * X + b

sqrloss = tf.square(Y - Y_pred, name='loss')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(sqrloss)

tic = time.time() # A suspicious naming style from MATLAB, which must indicate sth.

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        loss = 0
        for x, y in data:
            _, sqrloss_ = sess.run([optimizer, sqrloss], feed_dict={X: x, Y:y})
            loss += sqrloss_
            pass
        print('Epoch {0}: {1}'.format(i, loss/len_data))
    w_out, b_out = sess.run([w, b])
    writer.close()

toc = time.time()
print('Took: %f seconds' %(toc - tic))



plt.plot(data[:,0], data[:,1], 'b.', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data', linewidth=1)
plt.legend()
plt.show()