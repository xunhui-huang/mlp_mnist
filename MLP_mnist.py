# -*- coding: utf-8 -*-
import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def parse():
    parse=argparse.ArgumentParser('Super parameters')
    parse.add_argument('-B',dest='BATCH_SIZE',default=100,required=False)
    parse.add_argument('-N',dest='NUM_EPOCHS',default=3000,required=False)
    parse.add_argument('-k_tr',dest='train_keep_prob',default=0.75,required=False)
    parse.add_argument('-k_te',dest='test_keep_prob',default=1.0,required=False)
    parse.add_argument('-l',dest='learning_rate',default=0.005,required=False)
    args=parse.parse_args()
    return args

args=parse()
BATCH_SIZE=args.BATCH_SIZE
NUM_EPOCHS=args.NUM_EPOCHS
train_keep_prob=args.train_keep_prob
test_keep_prob=args.test_keep_prob
lr=args.learning_rate

mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

x=tf.placeholder(shape=[None,784],dtype=tf.float32,name='x')
y=tf.placeholder(shape=[None,10],dtype=tf.float32,name='y')

def inference(x,keep_prob):
    with tf.variable_scope('hidden_layer',reuse=tf.AUTO_REUSE):
        w1=tf.get_variable('weights1',shape=[784,100],dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,1))
        b1=tf.get_variable('bias1',dtype=tf.float32,initializer=tf.zeros([100]))
        hidden1=tf.nn.relu(tf.matmul(x,w1)+b1)
        hidden1_drop=tf.nn.dropout(hidden1,keep_prob)
    with tf.variable_scope('output_layer',reuse=tf.AUTO_REUSE):
        w2=tf.get_variable('weights2',shape=[100,10],dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,1))
        b2=tf.get_variable('bias2',dtype=tf.float32,initializer=tf.zeros([10]))
        output=tf.nn.softmax(tf.matmul(hidden1_drop,w2)+b2)
    return output

train_y_pred=inference(x,train_keep_prob)
train_cross_entropy=tf.reduce_mean(-tf.reduce_sum(y*tf.log(tf.clip_by_value(train_y_pred,0.0001,1))),name='train_cross_entropy')
tf.get_variable_scope().reuse_variables()
test_y_pred=inference(x,test_keep_prob)
test_cross_entropy=tf.reduce_mean(-tf.reduce_sum(y*tf.log(tf.clip_by_value(test_y_pred,0.0001,1))))
correct_prediction=tf.equal(tf.argmax(test_y_pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tf.summary.scalar('train_cross_entropy',train_cross_entropy)
merged_summary=tf.summary.merge_all()
summary_writer=tf.summary.FileWriter('./mlp_mnist')
summary_writer.add_graph(tf.get_default_graph())

train_op=tf.train.GradientDescentOptimizer(lr).minimize(train_cross_entropy)

init_op=tf.global_variables_initializer()
local_init_op=tf.local_variables_initializer()

gpu_options=tf.GPUOptions(allow_growth=True)
config=tf.ConfigProto(gpu_options=gpu_options)

with tf.Session(config=config) as sess:
    sess.run(init_op)
    sess.run(local_init_op)
    for i in range(NUM_EPOCHS):
        train_batch_x,train_batch_y=mnist.train.next_batch(BATCH_SIZE)
        train_loss,_,summary=sess.run([train_cross_entropy,train_op,merged_summary],feed_dict={x:train_batch_x,y:train_batch_y})
        summary_writer.add_summary(summary,i)
        if i %500==0:
            test_batch_x,test_batch_y=mnist.test.next_batch(BATCH_SIZE)
            test_loss,accu=sess.run([test_cross_entropy,accuracy],feed_dict={x:test_batch_x,y:test_batch_y})
            print('epoch %d,test_loss is:%f' % (i,test_loss))
            print('epoch %d,test_accuracy is %f' % (i,accu))
    
    