#-*- coding: utf-8 -*-
'''ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
'''

import numpy as np
import os,sys,inspect
import tensorflow as tf
import time

from input import Dataset
import globals as g_
from models import ResNet50

from keras.optimizers import SGD
from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing import image
import keras.backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
#from imagenet_utils import decode_predictions, preprocess_input
from keras.layers.core import Reshape,Lambda


def create_model():
	#model_perception = Sequential()
    image =Input(shape=(224,224,3))
    model_perception = ResNet50(include_top=True, weights='imagenet')(image)
    model_perception = Flatten()(model_perception)
    model_perception = Dense(12, activation='softmax',name='fc480')(model_perception)
    #sensor_output = Reshape((12,40,2))(model_perception)
    #reward_input = Lambda(lambda x : x[:,:,:,1])(sensor_output)
    #print reward_input.shape
    #sensor_output = Lambda(lambda x : x[:,:,:,0],  output_shape=lambda s: (s[0], s[1],s[2],1))(sensor_output)
    #reward_input = Input(shape=(12,40,1), name='reward_map')
    #value_map = Convolution2D(1,1,1, name='conv_q')(sensor_output)
    #for k in range(5):
#		x = merge([value_map, reward_input], mode='concat')
#		x = Convolution2D(12, 1,1, border_mode='same', name='conv_q'+str(k))(x)
#		value_map = Lambda(lambda x: K.max(x, axis=3,keepdims=True), output_shape=lambda s: (s[0], s[1],s[2],1))(x)
    
 #   value_map = Flatten()(value_map)
 #   actions = Dense(12, activation='softmax',name='fc_actions')(value_map)
#    model = Model(input=[image], output=[sensor_output,actions])
    model = Model(input=[image], output=[model_perception])
    print(model.summary())
    return model 

def main(argv):
    st = time.time()
    listfiles_train, labels_train = read_lists(g_.TRAIN_LOL)
    listfiles_val, labels_val = read_lists(g_.VAL_LOL)
    dataset_train = Dataset(listfiles_train, labels_train, subtract_mean=False, V=g_.NUM_VIEWS)
    dataset_val = Dataset(listfiles_val, labels_val, subtract_mean=False, V=g_.NUM_VIEWS)

    model = create_model()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

    #train(dataset_train, dataset_val, FLAGS.weights, FLAGS.caffemodel)
    #model.load_weights("model.h5")
    for epoch in range(100):
        final_loss = 0
        acc = 0
        num = 0
        acc_train = 0
        num_test = 0
        print('epoch:', epoch)
        dataset_train.shuffle()
        for batch_x, batch_y in dataset_train.batches(1):
            #print(batch_y)
            loss,acc1 = model.train_on_batch(batch_x, batch_y)
            acc_train += acc1
            final_loss += loss
            num += 1

        print('loss: ',final_loss/num)
        for batch_x, batch_y in dataset_val.batches(1):
            #print(batch_y)
            loss,acc2 = model.test_on_batch(batch_x, batch_y)
            acc += acc2
            num_test += 1
        print('acc_train: ', acc_train/num)
        print('acc_test: ', acc/num_test)
        pathmodel = "temp/model_%s.h5" % (epoch)
        model.save(pathmodel)
        print("Saved model to disk")	    


def read_lists(list_of_lists_file):
    listfile_labels = []

    text_file = open(list_of_lists_file, 'r')
    for line in text_file:
        file, label = line.split()
        listfile_labels.append((file, int(label)))

    listfiles, labels = zip(*listfile_labels)

    return listfiles, labels
    
if __name__ == '__main__':
    main(sys.argv)


