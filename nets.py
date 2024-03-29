from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
import numpy as np

def simple_cnn(input_shape, n_classes, initial_bias=None):
    net = tf.keras.models.Sequential()
    net.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=input_shape))
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(Flatten())
    net.add(Dense(units=64, activation='relu'))

    if initial_bias is None:
        #default initial bias is zeroes
        initial_bias = tf.constant_initializer([0 for i in range(n_classes)])
    net.add(Dense(units=n_classes, activation='softmax', bias_initializer=initial_bias))
    net.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return net

def vgg16_net_small_with_regularisation(_input_shape, n_classes, reg_dropout_rate=0, reg_wdecay_beta=0, reg_batch_norm=0):
    net = tf.keras.models.Sequential()
    net.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=_input_shape ))
    net.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    if reg_batch_norm: net.add(BatchNormalization())
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    if reg_batch_norm: net.add(BatchNormalization())
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    if reg_batch_norm: net.add(BatchNormalization())
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    if reg_batch_norm: net.add(BatchNormalization())
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    if reg_batch_norm: net.add(BatchNormalization())
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(Flatten())
    if reg_wdecay_beta > 0:
        reg_wdecay = tf.keras.regularizers.l2(reg_wdecay_beta)
    else:
        reg_wdecay = None
    net.add(Dense(units=256, activation='relu',kernel_regularizer=reg_wdecay))
    net.add(Dense(units=256, activation='relu',kernel_regularizer=reg_wdecay))
    net.add(Dense(units=128, activation='relu',kernel_regularizer=reg_wdecay))
    if reg_dropout_rate > 0: 
        net.add(Dropout(reg_dropout_rate))
    net.add(Dense(units=n_classes, activation='softmax'))
    return net

def vgg11_net(_input_shape, n_classes):
    net = tf.keras.models.Sequential()
    net.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=_input_shape ))
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(Flatten())
    net.add(Dense(units=1024, activation='relu'))
    net.add(Dense(units=1024, activation='relu'))
    net.add(Dense(units=512, activation='relu'))
    net.add(Dense(units=n_classes, activation='softmax'))
    return net

def vgg16_net_small(_input_shape, n_classes):
    net = tf.keras.models.Sequential()
    net.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=_input_shape ))
    net.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(Flatten())
    net.add(Dense(units=256, activation='relu'))
    net.add(Dense(units=256, activation='relu'))
    net.add(Dense(units=128, activation='relu'))
    net.add(Dense(units=n_classes, activation='softmax'))
    return net


def example4_net(_input_shape, n_classes, reg_dropout_rate=0, reg_wdecay_beta=0, reg_batch_norm=0):
    net = tf.keras.models.Sequential()
    net.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=_input_shape ))
    net.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    if reg_batch_norm: net.add(BatchNormalization())
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    if reg_batch_norm: net.add(BatchNormalization())
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    net.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    if reg_batch_norm: net.add(BatchNormalization())
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    net.add(Flatten())
    if reg_wdecay_beta > 0:
        reg_wdecay = tf.keras.regularizers.l2(reg_wdecay_beta)
    else:
        reg_wdecay = None
    net.add(Dense(units=128, activation='relu', kernel_regularizer=reg_wdecay))
    net.add(Dense(units=512, activation='relu', kernel_regularizer=reg_wdecay))
    if reg_dropout_rate > 0: 
        net.add(Dropout(reg_dropout_rate))
    net.add(Dense(units=n_classes, activation='softmax'))

    return net

