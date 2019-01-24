import util
import numpy as np 
import tensorflow as tf 
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 30
n_train = 60000
n_test = 10000

mnist_folder = '/home/eralien/storage/DataStorage/data/mnist'
util.download_mnist(mnist_folder)
train, val, test = util.read_mnist(mnist_folder, flatten=True)

# train is 55000*784, val is 5000*784, test is ?*784
train_data = tf.contrib.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000)
train_data = train_data.batch(batch_size)

test_data = tf.contrib.data.Dataset.from_tensor_slices(test)
test_data = test_data.shuffle(10000)
test_data = test_data.batch(batch_size)

# With dataset.make_initializable_iterator() with only make for one dataset
# like training set; then we have to make another subgraph for test set
# so we make a from_structure iterator with make_initializer
# for a multiple initialization
iterator = tf.contrib.data.Iterator.from_structure(train_data.output_types,
                                                train_data.output_shapes)
img, label = iterator.get_next() # img is 1*784, label is 1*10

train_init = iterator.make_initializer(train_data)
test_init  = iterator.make_initializer(test_data)

w = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.random_normal_initializer(stddev=0.01))
b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())

logits = tf.matmul(img, w) + b # which means logits.shape = (1, 10)

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label)
loss = tf.reduce_mean(entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# get accuracy
pred = tf.nn.softmax(logits) # pred.shape = (1, 10)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(label, 1))
# argmax(*, 1) returns the index of the biggest in columns,
# which is to check if any entry in pred is 1/0, and then compare with the label
# then, if equal: True, else: False.

# reduce_sum = sum(), with different dimension settings.
# tf.cast is to change the data type from tf.int32 to tf.float32
accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
with tf.Session() as sess:
    tic = time.time()
    sess.run(tf.global_variables_initializer())

    for i in range(n_epochs):
        sess.run(train_init)
        total_loss = 0
        n_batch = 0
        try:
            while True:
                _, loss_ = sess.run([optimizer, loss])
                total_loss += loss_
                n_batch += 1
        except tf.errors.OutOfRangeError:
            pass
        
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batch))

        sess.run(test_init)
        total_correct_pred = 0

        try:
            while True:
                accurate_batch = sess.run(accuracy)
                total_correct_pred += accurate_batch
        except tf.errors.OutOfRangeError:
            pass
        
        print('Accuracy {0}'.format(total_correct_pred/n_test))
writer.close()
