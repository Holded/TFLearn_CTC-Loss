from __future__ import division, print_function, absolute_import
import tflearn
import speech_data
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tflearn.layers.recurrent import BasicLSTMCell

from tensorflow.python.framework import sparse_tensor

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
init = tf.global_variables_initializer()
session.run(init)

learning_rate = 0.001
training_iters = 3000  # steps
batch_size = 16
total_size = 2400

width = 21  # mfcc+delta+delta^2 features
height = 320  # (max) length of utterance
classes = 10  # digits

batch = speech_data.mfcc_batch_generator(total_size, height)
X, Y = next(batch)

indices = tf.where(tf.not_equal(tf.cast(np.array(Y),tf.float32),0))
Y = tf.SparseTensor(indices, values=tf.gather_nd(Y,indices)-1, dense_shape=tf.cast(tf.shape(Y),tf.int64))

if isinstance(Y, sparse_tensor.SparseTensor):
  print("Good Job!")
else:
  print("This is not a SparseTensor_ from demo.py")

# Network building
net = tflearn.input_data([None, width, height])
net = tflearn.lstm(net, 50, dropout=0.8, return_seq=True)
net = tflearn.lstm(net, 50, dropout=0.8, return_seq=True)
net = tflearn.lstm(net, 50, dropout=0.8)
net = tflearn.fully_connected(net, classes)#, activation='softmax')

adam= tflearn.optimizers.adam(learning_rate=0.001,beta1=0.99,beta2=0.999)
net = tflearn.regression(net, optimizer=adam, loss='ctc_loss')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)  #best_checkpoint_path='./best_checkpoint/',best_val_accuracy=0.23)

model.fit(X, Y, n_epoch=training_iters, validation_set=0.1, show_metric=True,batch_size=batch_size,run_id='run_id_liqin')
model.save("tflearn.lstm.model")
