import pandas as pd
import tensorflow as tf
import os
import shutil

log_dir = 'summary/graph2/'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir)
print('created log_dir path')

data = pd.read_csv('glass.data.txt', header=None)
del data[0]

#normalization
for i in range(1, 10):
    data[i] = (data[i] - data[i].mean())/data[i].std()

data = data.sample(frac=1)
target = data.pop(10)
target = pd.get_dummies(target)

train_data = data[:190]
val_data = data[190:]
target_train = target[:190]
target_val = target[190:]

feat_train = train_data.to_numpy()
feat_val = val_data.to_numpy()
target_train = target_train.to_numpy()
target_val = target_val.to_numpy()

xt = tf.placeholder(dtype=tf.float32, shape=[None, 9], name='features')
yt = tf.placeholder(dtype=tf.float32, shape=[None, 6], name='labels')


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.truncated_normal([1, out_size], mean=-0.5, stddev=1.0), name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('regularization'):
            reg = tf.nn.l2_loss(Weights)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs, reg

h1 = 8
h2 = 8

layer1, reg1 = add_layer(xt, 9, h1, n_layer=1, activation_function=tf.nn.relu)
layer2, reg2 = add_layer(layer1, h1, h2, n_layer=2, activation_function=tf.nn.relu)
prediction, reg3 = add_layer(layer2, h2, 6, n_layer=3, activation_function=tf.nn.softmax)

with tf.name_scope('cost'):
    cost = (tf.losses.softmax_cross_entropy(onehot_labels=yt, logits=prediction, scope='Cost_Function')
            + 0.01*reg1 + 0.01*reg2 + 0.01*reg3)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.arg_max(prediction, 1, name='Argmax_Pred'),tf.arg_max(yt, 1, name='Y_Pred')
                                  ,name='train_match')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32, name='Cast_Error_Pred'), name='Accuracy')

with tf.name_scope('optimizaton'):
    optimizator = tf.train.AdagradOptimizer(learning_rate=0.07, name='optimizer').minimize(cost)

init = tf.global_variables_initializer()
cost1 = tf.summary.scalar('Loss', cost)
accuracy1 = tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test', sess.graph)
    sess.run(init)

    for step in range(1500):
        sess.run(optimizator, feed_dict={xt: feat_train, yt: target_train})
        train_vis = sess.run(merged, feed_dict={xt: feat_train, yt: target_train})
        val_vis1, val_vis2 = sess.run([cost1, accuracy1], feed_dict={xt: feat_val, yt: target_val})
        train_writer.add_summary(train_vis, step)
        test_writer.add_summary(val_vis1, step)
        test_writer.add_summary(val_vis2, step)

#Open tensorboard
#tensorboard --logdir summary/graph2
