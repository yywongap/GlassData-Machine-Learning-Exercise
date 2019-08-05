import tensorflow as tf
import pandas as pd
import os
import shutil

config  = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

log_dir = 'summary/graph2/'
if os.path.exists(log_dir):   # 删掉以前的summary，以免重合
    shutil.rmtree(log_dir)
os.makedirs(log_dir)
print('created log_dir path')

data = pd.read_csv('agaricus-lepiota.data', header=None)
del data[11]
Colname = {0:'Class', 1:'cap-shape', 2:'cap-surface', 3:'cap-color', 4:'bruises',
             5:'odor', 6:'gill-attachment', 7:'gill-spacing', 8:'gill-size',
             9:'gill-color', 10:'stalk-shape', 12:'stalk-surface-above-ring',
             13:'stalk-surface-below-ring', 14:'stalk-color-above-ring',
             15:'stalk-color-below-ring', 16:'veil-type',17:'veil-color',
             18:'ring-number', 19:'ring-type', 20:'spore-print-color',
             21:'population',22:'habitat'}
data.rename(Colname, axis='columns', inplace=True)
data = pd.get_dummies(data, prefix=([v for i, v in Colname.items()]),drop_first=True)
data = data.sample(frac=1)
#label = pd.concat([data.pop('Class_e'),data.pop('Class_p')], axis=1,sort=False)
#feature = data
label, feature = data.pop('Class_p'), data
#print(data.head())
#print(list(label))
train_label, train_feature = label[:6000, None], feature[:6000]
val_label, val_feature = label[6000:7500, None], feature[6000:7500]
test_label, test_feature = label[7500:, None], feature[7500:]
#print(train_label.shape)

#placeholder
x = tf.placeholder(dtype=tf.float32, shape=[None, 91], name='features')
y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='labels')
#y = tf.cast(y,dtype=tf.float32)

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.truncated_normal([1, out_size], mean=-0.5, stddev=1.0), name='b')
            #tf.zeros([1, out_size])
            tf.summary.histogram(layer_name + '/biases', biases)
        #with tf.name_scope('regularization'):
         #   reg = tf.nn.l2_loss(Weights)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
            #  mathematically, Xw + b ,in python code - tf.matmul(X, w)
            #  with x having sample as rows, features as columns
            #  Also with w having features as row, nodes as columns
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs, Wx_plus_b

outsize1 = 8
outsize2 = 8
outsize3 = 8

layer1, W1 = add_layer(inputs=x, in_size=91, out_size=outsize1, n_layer=1, activation_function=tf.nn.relu)
layer2, W2 = add_layer(inputs=layer1, in_size=outsize1, out_size=outsize2, n_layer=2, activation_function=tf.nn.relu)
layer3, W3 = add_layer(inputs=layer2, in_size=outsize2, out_size=outsize3, n_layer=3, activation_function=tf.nn.relu)
prediction, logit = add_layer(inputs=layer3, in_size=outsize3, out_size=1, n_layer=4, activation_function=tf.nn.sigmoid)

with tf.name_scope('cost'):

    cost = tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=logit, scope='Cost_Function')
    #cost = (tf.losses.softmax_cross_entropy(onehot_labels=, logits=prediction, scope='Cost_Function')
     #       + 0.01*reg1 + 0.01*reg2 + 0.01*reg3)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(tf.concat([prediction, tf.math.add(1., -prediction)],axis=1),1, name='Argmax_Pred')
                                  ,tf.argmax(tf.concat([y,tf.math.add(1., -y)],axis=1),1, name='Y_Pred'),name='train_match')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32), name='Accuracy')

with tf.name_scope('optimization'):
    optimizator = tf.train.AdagradOptimizer(learning_rate=0.07, name='optimizer').minimize(cost)

init = tf.global_variables_initializer()
cost1 = tf.summary.scalar('Loss', cost)
accuracy1 = tf.summary.scalar('Accuracy',accuracy)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    val_writer = tf.summary.FileWriter(log_dir + '/val', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test', sess.graph)
    sess.run(init)

    for step in range(2000):
        sess.run(optimizator, feed_dict={x: train_feature, y: train_label})
        train_vis = sess.run(merged, feed_dict={x: train_feature, y: train_label})
        #val_vis = sess.run(merged, feed_dict={xt: feat_val, yt: target_val})
        val_vis1, val_vis2 = sess.run([cost1, accuracy1], feed_dict={x: val_feature, y: val_label})
        train_writer.add_summary(train_vis, step)
        #test_writer.add_summary(val_vis, step)
        val_writer.add_summary(val_vis1, step)
        val_writer.add_summary(val_vis2, step)

    #print(sess.run(prediction, feed_dict={x: val_feature, y: val_label}))
    test_vis = sess.run(accuracy1, feed_dict={x: test_feature, y: test_label})
    test_writer.add_summary(test_vis, step)
    print('The test set accuracy is', sess.run(accuracy, feed_dict={x: test_feature, y: test_label}))

'''
open tensorboard 
tensorboard --logdir summary/graph2

Set classification threshold as 50%.
The train accuracy is about 98%.
the validation set accuracy is about 99%.
The test set accuracy is about 98%.
'''