import tensorflow as tf 
import os
# import shutil

tf.set_random_seed(0)
cwd = os.getcwd()
graph_dir = cwd + '/graphs'
filelist = [ f for f in os.listdir(graph_dir) if f.endswith(".eralienPC") ]
for f in filelist:
    os.remove(os.path.join(graph_dir, f))
# shutil.rmtree(cwd + "/graphs")



with tf.device('/gpu:0'):
    x = tf.constant([[1.0, 1.1, 1.2, 1.3, 1.4, 1.5], [0, 0.1, 0.2, 0.3, 0.4, 0.5]], name='tf_x')
    y = tf.constant([[1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 1, 1]], dtype='float32', name='tf_y')
    z = tf.multiply(x, y, name='multi1')
    to1_0s0 = tf.zeros([2, 3], tf.int32)
    to2_0s1 = tf.zeros_like(y)
    to3_1s0 = tf.ones([2, 3], tf.int32)
    to4_1s1 = tf.ones_like(y)
    to5_1p5s = tf.fill([2, 3], 1.5, name='1.5s')
    to6_lin0 = tf.lin_space(0.0, 5.0, 6)
    to7_rg0 = tf.range(0.0, 2.2, 0.5)
to = [to1_0s0, to2_0s1, to3_1s0, to4_1s1, to5_1p5s, to6_lin0, to7_rg0]
with tf.Session() as sess0:
    to1_0s0, to2_0s1, to3_1s0, to4_1s1, to5_1p5s, to6_lin0, to7_rg0 = sess0.run(to)
    to = [to1_0s0, to2_0s1, to3_1s0, to4_1s1, to5_1p5s, to6_lin0, to7_rg0]
    i = 0
    for to_item in range(len(to)):
        print("to" + str(to_item + 1) + " = \n" + str(to[to_item]) + '\n')

"""Sess1"""
# print("to1 = " + str(to1_0s0)+ "\nto2 = " + str(to2_0s1))
op1 = tf.add(x, y, name='add1')
op2 = tf.multiply(x, y, name='add2')
op3 = tf.pow(op1, op2, name='pow1')
op4 = tf.multiply(op2, op3, name='multi2')
with tf.Session() as sess1:
    op3, op4 = sess1.run([op3, op4])
    print("op3 = \n" + str(op3)+ "\nop4 = \n" + str(op4))
pass

# create variables with class Variable
s_0 = tf.Variable(2, name="scalar0")
m_0 = tf.Variable(to5_1p5s, name="matrix0")
W_0 = tf.Variable(tf.ones([700,10]), name="big_matrix0")
N_0 = tf.Variable(tf.truncated_normal([700,10]))

# create variables with func get_variable
s_1 = tf.get_variable("scalar1", initializer=tf.constant(2))
m_1 = tf.get_variable("matrix1", initializer=to5_1p5s)
W_1 = tf.get_variable("big_matrix1", shape=(700, 10), initializer=tf.zeros_initializer())

# create an assignment to N_0
N_0_multi_W_1_op = N_0.assign(tf.multiply(N_0, W_1))
N_0_add_W_0_op = N_0.assign(tf.add(N_0, W_0))

# Variable class obj cannot start Session without an initialization
# Initialization goes this way:
with tf.Session() as sess2:
    sess2.run(tf.variables_initializer([s_0, m_0])) # To initialize a set of certain variable
    sess2.run(W_0.initializer) # To initialize one certain variable
    sess2.run(tf.global_variables_initializer()) # This command initialize all variables at once, use this one!!!
    print("\nN_0 before op:\n\n" + str(N_0.eval())) # >> display normal distributed array 

    # Do the op
    sess2.run(N_0_multi_W_1_op)
    print(W_0) # >> <tf.Variable 'big_matrix0:0' shape=(700, 3) dtype=float32_ref>
    print("\nN_0 after op:\n\n" + str(N_0.eval())) # >> display real zeros as array

    # Control op sequence
    get = tf.get_default_graph()
    # with get.control_dependencies([N_0_add_W_0_op,])
    writer = tf.summary.FileWriter('./graphs', get)
# This initialize an op. We need to execute these orders w/i a context of this session

pass


a = tf.placeholder(tf.float32, shape=[3,4])
b = tf.truncated_normal([3,4], mean=2, stddev=4)
c = a + b

# with tf.Session() as sess:
#     c = sess.run(c, feed_dict={a: np.ones([3,4])})
#     print(c)
#     writer = tf.summary.FileWriter('./graphs', get)


x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('./graphs/lazy_loading', sess.graph)
	for _ in range(10):
		sess.run(x + y)
	# print(tf.get_default_graph().as_graph_def()) 
	writer.close()

# writer.close()

writer.close()
sess0.close()
sess1.close()
sess2.close()

tensorboard_flag = 0
if tensorboard_flag:
    # import Tensorboard_run
    pass


