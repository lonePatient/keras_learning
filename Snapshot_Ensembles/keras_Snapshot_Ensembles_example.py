#encoding:utf-8
'''
使用方式：
Usage
1.Training

    from mniny_inception_module import train

    run = 0
    while True:
        train(run)
        run += 1

2.Evaluation
ensemble

    from mniny_inception_module import evaluate_ensemble

    # To evaluate ensemble of all models in weights folder:
    evaluate_ensemble(Best=False)

    # To evaluate ensemble of best models per training session:
    evaluate_ensemble()

individual

    from mniny_inception_module import evaluate

    #Evaluate all models in weights directory:
    evaluate(eval_all=True)

    # Evaluate 'Best' models in weights directory:
    evaluate()

'''


import os
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,Merge,AveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.models import Model,load_model
from keras.layers.core import Dropout,Lambda,Reshape,Activation
from keras import backend as K
from keras.utils import np_utils
from keras.layers.advanced_activations import PReLU
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from snapshot import SnapshotCallbackBuilder
from keras import metrics
def miny_inception_module(x,scale = 1,predict = False):
    x11 = Conv2D(filters = int(16*scale),kernel_size=(1,1),strides=(1,1),padding='valid')(x)
    x11 = BatchNormalization()(x11)
    x11 = Activation('relu')(x11)

    x33 = Conv2D(filters = int(24*scale),kernel_size=(1,1),strides=(1,1),padding='valid')(x)
    x33 = BatchNormalization()(x33)
    x33 = Activation('relu')(x33)
    x33 = Conv2D(filters = int(32*scale),kernel_size=(3,3),strides=(1,1),padding='same')(x33)
    x33 = BatchNormalization()(x33)
    x33 = Activation('relu')(x33)

    x55 = Conv2D(filters=int(4*scale),kernel_size=(1,1),strides=(1,1),padding='valid')(x)
    x55 = BatchNormalization()(x55)
    x55 = Activation('relu')(x55)
    x55 = Conv2D(filters = int(8*scale),kernel_size=(5,5),strides=(1,1),padding='same')(x55)
    x55 = BatchNormalization()(x55)
    x55 = Activation("relu")(x55)

    x33p = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
    x33p = Conv2D(filters = int(8*scale),kernel_size=(1,1),strides=(1,1),paddding='valid')(x33p)
    x33p = BatchNormalization()(x33p)
    x33p = Activation('relu')(x33p)
    out = Merge(layers=[x11,x33,x55,x33p],mode='concat',concat_axis=3)
    if predict:
        prediction = AveragePooling2D(pool_size=(5,5),strides=(1,1),padding='valid')(x)
        prediction = Conv2D(filters=int(8*scale),kernel_size=(1,1),strides=(1,1))(prediction)
        prediction = BatchNormalization()(prediction)
        prediction = Activation('relu')(prediction)
        prediction = Dropout(0.25)(prediction)
        prediction =Flatten()(prediction)
        prediction = Dense(120)(prediction)
        prediction = BatchNormalization()(prediction)
        prediction =Activation("relu")(prediction)
        predict = Dense(10,activation='softmax')(prediction)
        return out ,predict
    return out


def inception_net(_input):
    x = Reshape((28,28,1))(_input)
    x =Conv2D(filters = 16,kernel_size=(3,3),strides=(2,2),padding='valid',name='conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Activation(PReLU())(x)
    x = Conv2D(filters = 48,kernel_size=(3,3),strides=(1,1),paddding='valid',name='conv2')(x)
    x =BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters = 48,kernel_size=(3,3),strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = miny_inception_module(x,1)
    x = miny_inception_module(x,2)
    x = miny_inception_module(x,2)
    x,soft1 = miny_inception_module(x,4,True)

    x = miny_inception_module(x,3)
    x = miny_inception_module(x,3)
    x,soft2 = miny_inception_module(x,4,True)

    x = MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)
    x = miny_inception_module(x,4)
    x = miny_inception_module(x,5)
    x = AveragePooling2D(pool_size=(5,5),strides=(1,1))(x)
    x = Dropout(0.4)(x)
    x =Flatten()(x)
    soft3 = Dense(10,activation='softmax')(x)
    out = Merge(layers=[soft1,soft2,soft3],mode= 'avg',concat_axis=1)
    return out

def create_model():
    _input = Input((784,))
    incep1 = inception_net(_input)
    out = incep1
    model = Model(inputs=_input,outputs=[out])
    return model

def train(run = 0):
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    x_train = x_train.reshape(60000,784)
    x_test = x_test.reshape(10000,784)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    # datagen = ImageDataGenerator(rotation_range=45,width_shift_range=0.2,height_shift_range=0.2)
    # datagen.fit(x_train)

    y_train = np_utils.to_categorical(y_train,num_classes=10)
    y_test  = np_utils.to_categorical(y_test,num_classes=10)

    model = create_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    nb_epoch = 80
    nb_musicians = 5
    snapshot = SnapshotCallbackBuilder(nb_epoch, nb_musicians, init_lr=0.006)
    # model.fit(datagen.flow(x_train,y_train,batch_size=32),epochs = nb_epoch)
    model.fit(x_train,y_train,batch_size=1024,epochs = nb_epoch,
              verbose=1,validation_data=(x_test,y_test),
              callbacks=snapshot.get_callbacks("snap-model"+str(run)))

    model = load_model("weights/%s-Best.h5"%("snap-model"+str(run)))
    model.compile(loss = 'categorial_crossentropy',optimizer='adam',metrics=['categorial_accuracy'])
    score = model.evaluate(x_test,y_test,verbose=0)

    print '--------------------------------------'
    print 'model'+str(run)+':'
    print 'Test loss:', score[0]
    print 'error:', str((1.-score[1])*100)+'%'
    return score

def test_model():
    "Test to ensure model compiles successfully"
    model = create_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print("MODEL COMPILES SUCCESSFULLY")

def evaluate_ensemble(Best=True):
    '''
    creates and evaluates an ensemble from the models in the model folder.
    '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(10000, 784)
    X_test = X_test.astype('float32')
    X_test /= 255
    Y_test = np_utils.to_categorical(y_test, 10)

    model_dirs = []
    # 这里是保存所有模型的路径
    for i in os.listdir('weights'):
        if '.h5' in i:
            if not Best:
                model_dirs.append(i)
            else:
                if 'Best' in i:
                    model_dirs.append(i)

    preds = []
    model = create_model()
    for mfile in model_dirs:
        print(os.path.join('weights',mfile))
        model.load_weights(os.path.join('weights',mfile))
        yPreds = model.predict(X_test, batch_size=128, verbose=1)
        preds.append(yPreds)

    weighted_predictions = np.zeros((X_test.shape[0], 10), dtype='float64')
    weight = 1./len(preds)
    for prediction in preds:
        weighted_predictions += weight * prediction
    y_pred =weighted_predictions

    # Right now, categorical crossentropy & accuracy require tensor objects
    Y_test = tf.convert_to_tensor(Y_test)
    y_pred = tf.convert_to_tensor(y_pred)

    loss = metrics.categorical_crossentropy(Y_test, y_pred)
    acc = metrics.categorical_accuracy(Y_test, y_pred)
    sess = tf.Session()
    print('--------------------------------------')
    print('ensemble')
    print('Test loss:', loss.eval(session=sess))
    print('error:', str((1.-acc.eval(session=sess))*100)+'%')
    print('--------------------------------------')

def evaluate(eval_all=False):
    '''
    evaluate models in the weights directory,
    defaults to only models with 'Best'
    '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(10000, 784)
    X_test = X_test.astype('float32')
    X_test /= 255
    Y_test = np_utils.to_categorical(y_test, 10)
    evaluations = []

    for i in os.listdir('weights'):
        if '.h5' in i:
            if eval_all:
                evaluations.append(i)
            else:
                if 'Best' in i:
                    evaluations.append(i)
    print(evaluations)
    model = create_model()
    for run, i in enumerate(evaluations):
        model.load_weights(os.path.join('weights',i))
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                    metrics=['categorical_accuracy'])
        score = model.evaluate(X_test, Y_test,
                            verbose=1)
        print('--------------------------------------')
        print('model'+str(run)+':')
        print('Test loss:', score[0])
        print('error:', str((1.-score[1])*100)+'%')