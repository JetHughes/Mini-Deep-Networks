__author__ = "Lech Szymanski"
__organization__ = "COSC420, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import show_methods
import os
import pickle
import gzip
from keras import metrics
from load_oxford_flowers102 import load_oxford_flowers102
import nets
from utils import balanced_accuracy

EPOCHS=150

def oxford102_trained_cnn(load_from_file=False, verbose=True,
                        reg_wdecay_beta=0, reg_dropout_rate=0, reg_batch_norm=False, data_aug=False):

    train_data, validation_data, test_data, class_names = load_oxford_flowers102(imsize=96, fine=False)

    y_hat_train = train_data["labels"]
    y_hat_test = test_data["labels"]
    x_train = train_data["images"]
    x_test = test_data["images"]
    x_valid = validation_data["images"]
    y_hat_valid = validation_data["labels"]

    # Create 'saved' folder if it doesn't exist
    if not os.path.isdir("saved"):
        os.mkdir('saved')

    # Specify the names of the save files
    save_name = os.path.join('saved', 'oxford102_rwd%.1e_rdp%.1f_rbn%d_daug%d' % (
        reg_wdecay_beta, reg_dropout_rate, int(reg_batch_norm), int(data_aug)))
    net_save_name = save_name + '_cnn_net.h5'
    checkpoint_save_name = save_name + '_cnn_net.chk.weights.h5'
    history_save_name = save_name + '_cnn_net.hist'

    n_classes = len(class_names)

    if load_from_file and os.path.isfile(net_save_name):
        # ***************************************************
        # * Loading previously trained neural network model *
        # ***************************************************

        # Load the model from file
        if verbose:
            print("Loading neural network from %s..." % net_save_name)
        net = tf.keras.models.load_model(net_save_name)

        # Load the training history - since it should have been created right after
        # saving the model
        if os.path.isfile(history_save_name):
            with gzip.open(history_save_name) as f:
                history = pickle.load(f)
        else:
            history = []
    else:
        # ************************************************
        # * Creating and training a neural network model *
        # ************************************************

        # net = example4_net((96, 96, 3), n_classes, reg_dropout_rate, reg_wdecay_beta, reg_batch_norm)
        net = nets.vgg16_net_small_with_regularisation((96, 96, 3), n_classes, reg_dropout_rate, reg_wdecay_beta, reg_batch_norm)
        # net = nets.vgg16_net_small((96, 96, 3), n_classes)

        # Define training regime: type of optimiser, loss function to optimise and type of error measure to report during
        # training
        # METRICS = [
        #     metrics.MeanSquaredError(name='Brier score'),
        #     metrics.TruePositives(name='tp'),
        #     metrics.FalsePositives(name='fp'),
        #     metrics.TrueNegatives(name='tn'),
        #     metrics.FalseNegatives(name='fn'), 
        #     metrics.BinaryAccuracy(name='accuracy'),
        #     metrics.Precision(name='precision'),
        #     metrics.Recall(name='recall'),
        #     metrics.AUC(name='auc'),
        #     metrics.AUC(name='prc', curve='PR'), # precision-recall curve
        # ]

        net.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        # Training callback to call on every epoch -- evaluates
        # the model and saves its weights if it performs better
        # (in terms of accuracy) on validation data than any model
        # from previous epochs
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_save_name,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        if data_aug:

            # Create data generator that randomly manipulates images
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                zca_epsilon=1e-06,
                width_shift_range=0.1,
                height_shift_range=0.1,
                fill_mode='nearest',
                horizontal_flip=True
            )

            # Configure the data generator for the images in the training sets
            datagen.fit(x_train)

            # Build the data generator
            train_data_aug = datagen.flow(x_train, y_hat_train)

            train_info = net.fit(train_data_aug,
                                 validation_data=(x_valid, y_hat_valid),
                                 epochs=EPOCHS, shuffle=True,
                                 callbacks=[model_checkpoint_callback])
        else:
            train_info = net.fit(x_train, y_hat_train, 
                                 validation_data=(x_valid, y_hat_valid),
                                 epochs=EPOCHS, shuffle=True,
                                 callbacks=[model_checkpoint_callback])

        # Load the weights of the best model
        print("Loading best save weight from %s..." % checkpoint_save_name)
        net.load_weights(checkpoint_save_name)

        # Save the entire model to file
        print("Saving neural network to %s..." % net_save_name)
        net.save(net_save_name)

        # Save training history to file
        history = train_info.history
        with gzip.open(history_save_name, 'w') as f:
            pickle.dump(history, f)

    # *********************************************************
    # * Training history *
    # *********************************************************

    # Plot training and validation accuracy over the course of training
    if verbose and history != []:
        fh = plt.figure()
        ph = fh.add_subplot(111)
        ph.plot(history['accuracy'], label='accuracy')
        ph.plot(history['val_accuracy'], label='val_accuracy')
        ph.set_xlabel('Epoch')
        ph.set_ylabel('Accuracy')
        ph.set_ylim([0, 1])
        ph.legend(loc='lower right')

    # *********************************************************
    # * Evaluating the neural network model within tensorflow *
    # *********************************************************

    if verbose:
        # Compute output for 16 test images
        y_pred = net.predict(x_test)

        loss_train, accuracy_train,  = net.evaluate(x_train, y_hat_train, verbose=0)
        loss_test, accuracy_test = net.evaluate(x_test, y_hat_test, verbose=0)

        print("Train accuracy (tf): %.2f" % accuracy_train)
        print("Test accuracy  (tf): %.2f" % accuracy_test)
        balanced_accuracy(test_labels=y_hat_test, predictions=y_pred, outputs=n_classes)

    net.summary()

    return net


if __name__ == "__main__":
    oxford102_trained_cnn(load_from_file=True, verbose=True,
                        reg_wdecay_beta=0.1, reg_dropout_rate=0.4, reg_batch_norm=True, data_aug=True)
