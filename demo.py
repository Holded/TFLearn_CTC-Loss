from __future__ import division, print_function, absolute_import
import tflearn
import speech_data
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tflearn.layers.recurrent import BasicLSTMCell

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

Y = speech_data.convert_to_sparse(Y)

# Network building
net = tflearn.input_data([None, width, height])
net = tflearn.lstm(net, 50, dropout=0.8, return_seq=True)
net = tflearn.lstm(net, 50, dropout=0.8, return_seq=True)
net = tflearn.lstm(net, 50, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')

adam= tflearn.optimizers.adam(learning_rate=0.001,beta1=0.99,beta2=0.999)
net = tflearn.regression(net, optimizer=adam, loss='ctc_loss')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)  #best_checkpoint_path='./best_checkpoint/',best_val_accuracy=0.23)

model.fit(X, Y, n_epoch=training_iters, validation_set=0.1, show_metric=True,batch_size=batch_size,run_id='run_id_liqin')
model.save("tflearn.lstm.model")
