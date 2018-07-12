"""Code to classify sounds for the Freesound General-Purpose Audio Tagging Challenge:
https://www.kaggle.com/c/freesound-audio-tagging.
Based on the notebook posed by Zafar:
https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data/notebook
Thanks Zafar!
"""

import os
import shutil
import pandas as pd
import librosa
import numpy as np
import keras
import GPyOpt
from sklearn.cross_validation import StratifiedKFold
from keras import losses, models, optimizers
from keras.activations import relu, softmax
import tensorflow as tf

np.random.seed(1001)

class SoundClassifier(object):
    def __init__(self):
        self.batch_size = 64
        self.dropout_prob = 0.1
        self.learning_rate = 0.001
        self.layer_group_1_kernel = 9
        self.layer_group_1_n_convs = 16
        self.layer_group_1_max_pool = 16
        self.layer_group_2_kernel = 3
        self.layer_group_2_n_convs = 32
        self.layer_group_2_max_pool = 4
        self.layer_group_3_kernel = 3
        self.layer_group_3_n_convs = 32
        self.layer_group_3_max_pool = 4
        self.layer_group_4_kernel = 3
        self.layer_group_4_n_convs = 256
        self.dense_1_n_hidden = 64
        self.dense_2_n_hidden = 1024
        self.use_mfcc = True
        self.best_accuracy = 0

    def set_params(self, values):
        params = ['batch_size', 'layer_group_1_kernel', 'layer_group_1_n_convs', 'layer_group_1_max_pool',
                  'layer_group_2_kernel', 'layer_group_2_n_convs', 'layer_group_2_max_pool', 'layer_group_3_kernel',
                  'layer_group_3_n_convs', 'layer_group_3_max_pool', 'layer_group_4_kernel', 'layer_group_4_n_convs',
                  'dense_1_n_hidden', 'dense_2_n_hidden', 'dropout_prob', 'learning_rate',
                  ]
        if self.use_mfcc:
            params = ['batch_size', 'layer_group_1_kernel', 'layer_group_1_n_convs',
                      'layer_group_2_kernel', 'layer_group_2_n_convs', 'layer_group_3_kernel',
                      'layer_group_3_n_convs', 'layer_group_4_kernel', 'layer_group_4_n_convs',
                      'dense_1_n_hidden', 'dropout_prob', 'learning_rate',
                      ]
        float_params = ["dropout_prob", "learning_rate"]
        for value, param in zip(values, params):
            if param not in float_params:
                value = int(value)
            setattr(self, param, value)
            print "setting {} to {}".format(param, value)

    def get_2d_conv_model_default(self, config=None, input_tensor=None, compile_model=False):
        if config:
            nclass = config.n_classes
            inp = keras.layers.Input(shape=(config.dim[0], config.dim[1], 1))
        else:
            nclass = 41
            inp = keras.layers.Input(tensor=input_tensor)
        x = keras.layers.Convolution2D(32, (4, 10), padding="same")(inp)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D()(x)

        x = keras.layers.Convolution2D(32, (4, 10), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D()(x)

        x = keras.layers.Convolution2D(32, (4, 10), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D()(x)

        x = keras.layers.Convolution2D(32, (4, 10), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D()(x)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(64)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        out = keras.layers.Dense(nclass, activation=softmax)(x)

        model = models.Model(inputs=inp, outputs=out)
        if compile_model:
            opt = optimizers.Adam(config.learning_rate)
            model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        return model

    def get_2d_conv_model(self, config=None, input_tensor=None, compile_model=False):
        if config:
            nclass = config.n_classes
            inp = keras.layers.Input(shape=(config.dim[0], config.dim[1], 1))
        else:
            nclass = 41
            inp = keras.layers.Input(tensor=input_tensor)
        x = keras.layers.Convolution2D(self.layer_group_1_n_convs,
                                       (self.layer_group_1_kernel, self.layer_group_1_kernel), padding="same")(inp)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D()(x)
        x = keras.layers.Dropout(rate=self.dropout_prob)(x)

        x = keras.layers.Convolution2D(self.layer_group_2_n_convs,
                                       (self.layer_group_2_kernel, self.layer_group_2_kernel), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D()(x)
        x = keras.layers.Dropout(rate=self.dropout_prob)(x)

        x = keras.layers.Convolution2D(self.layer_group_3_n_convs,
                                       (self.layer_group_3_kernel, self.layer_group_3_kernel), padding="same")(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D()(x)
        x = keras.layers.Dropout(rate=self.dropout_prob)(x)

        x = keras.layers.Convolution2D(self.layer_group_4_n_convs,
                                       (self.layer_group_4_kernel, self.layer_group_4_kernel), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D()(x)
        x = keras.layers.Dropout(rate=self.dropout_prob)(x)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(self.dense_1_n_hidden)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        out = keras.layers.Dense(nclass, activation=softmax)(x)

        model = models.Model(inputs=inp, outputs=out)
        if compile_model:
            opt = optimizers.Adam(config.learning_rate)
            model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        return model

    def label_parser(self, record):
        keys_to_features = {
            "label": tf.FixedLenFeature([], tf.int64)
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        # label = tf.decode_raw(parsed["label"], tf.int64)
        label = tf.one_hot(parsed["label"], 41)
        return label

    def feature_parser(self, record):
        keys_to_features = {
            "features": tf.FixedLenFeature([], tf.string)
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        features = tf.decode_raw(parsed["features"], tf.float64)
        features = tf.cast(features, tf.float16)
        features = tf.reshape(features, (40, 1 + int(np.floor(44100 * 2 / 512)), 1))
        return features

    def fn_parser(self, record):
        keys_to_features = {
            "filename": tf.FixedLenFeature([], tf.string)
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        #features = tf.decode_raw(parsed["features"], tf.string)
        fn = tf.cast(parsed["filename"], tf.string)
        # features = tf.reshape(features, (80, 1 + int(np.floor(24000 * 2 / 512)), 1))
        return fn

    def record_parse(self, record):
        keys_to_features = {
            "features": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.string)
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        features = tf.decode_raw(parsed["features"], tf.uint8)
        features = tf.cast(features, tf.float32)
        features = tf.reshape(features, (40, 1 + int(np.floor(44100 * 2 / 512)), 1))
        label = tf.decode_raw(parsed["label"], tf.float32)
        return features, label

    def train_tf_records(self, max_epochs=50):

        PREDICTION_FOLDER = "predictions_1d_conv"
        if not os.path.exists(PREDICTION_FOLDER):
            os.mkdir(PREDICTION_FOLDER)
        if os.path.exists('logs/' + PREDICTION_FOLDER):
            shutil.rmtree('logs/' + PREDICTION_FOLDER)

        checkpoint = keras.callbacks.ModelCheckpoint('best.h5', monitor='val_loss', verbose=1,
                                                     save_best_only=True)
        early = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5)
        tb = keras.callbacks.TensorBoard(log_dir='./logs/' + PREDICTION_FOLDER, write_graph=True)

        callbacks_list = [checkpoint, early, tb]
        print("#" * 50)

        train = pd.read_csv("./train.csv")
        test = pd.read_csv("./sample_submission.csv")
        LABELS = list(train.label.unique())
        label_idx = {label: i for i, label in enumerate(LABELS)}
        train.set_index("fname", inplace=True)
        test.set_index("fname", inplace=True)
        train["label_idx"] = train.label.apply(lambda x: label_idx[x])

        train_set_size = int(len(train["label_idx"]) * .9)
        test_set_size = int(len(train["label_idx"]) * .1)

        train_dataset = tf.data.TFRecordDataset(filenames=["./audio_40_mfcc_norm_train.tfrecords"])
        # Note: repeat before shuffle results in sampling with replacement
        # Also, not seeding the shuffle appears to cause the two iterators to fall out of synch.
        train_dataset = train_dataset.shuffle(train_set_size, seed=42).repeat()

        train_x = train_dataset.map(self.feature_parser)
        x_it = train_x.batch(self.batch_size).make_one_shot_iterator()

        train_y = train_dataset.map(self.label_parser)
        y_it = train_y.batch(self.batch_size).make_one_shot_iterator()
        model_train = self.get_2d_conv_model_default(input_tensor=x_it.get_next(), compile_model=False)
        opt = optimizers.Adam(self.learning_rate)
        model_train.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'],
                            target_tensors=[y_it.get_next()])
        test_dataset = tf.data.TFRecordDataset(filenames=["./audio_40_mfcc_norm_eval.tfrecords"]).repeat()
        test_x = test_dataset.map(self.feature_parser)
        x_it = test_x.batch(self.batch_size).make_one_shot_iterator()

        test_y = test_dataset.map(self.label_parser)
        y_it = test_y.batch(self.batch_size).make_one_shot_iterator()

        model_test = self.get_2d_conv_model_default(input_tensor=x_it.get_next(), compile_model=False)
        opt = optimizers.Adam(self.learning_rate)
        model_test.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'],
                           target_tensors=[y_it.get_next()])
        # TODO: Shuffle after each epoch
        for i in range(100):
            print "cycle {}".format(i)
            model_train.fit(steps_per_epoch=train_set_size/self.batch_size, callbacks=callbacks_list)
            model_train.save_weights("model.h5")

            model_test.load_weights("model.h5")
            accuracy = model_test.evaluate(steps=test_set_size/self.batch_size)[1]
            print "validation accuracy: {}".format(accuracy)
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy


def gpyopt_helper(x):
    """Objective function for GPyOpt.
    args:
        x: a 2D numpy array containing hyperparameters for the current acquisition
    returns:
        Error: The best test error for the training run."""

    sc = SoundClassifier()
    sc.set_params(x[0])
    sc.train_tf_records()
    # Convert accuracy to error
    error = 1 - sc.best_accuracy
    return np.array([[error]])


def bayes_opt():
    """Run bayesian optimization on the Sound Classifier using GPyOpt"""

    bounds = [{'name': 'batch_size', 'type': 'discrete', 'domain': range(64, 256)},
              {'name': 'layer_group_1_kernel', 'type': 'discrete', 'domain': range(4, 8)},
              {'name': 'layer_group_1_n_convs', 'type': 'discrete', 'domain': range(8, 64)},
              {'name': 'layer_group_1_max_pool', 'type': 'discrete', 'domain': range(4, 16)},
              {'name': 'layer_group_2_kernel', 'type': 'discrete', 'domain': range(8, 32)},
              {'name': 'layer_group_2_n_convs', 'type': 'discrete', 'domain': range(4, 64)},
              {'name': 'layer_group_2_max_pool', 'type': 'discrete', 'domain': range(4, 16)},
              {'name': 'layer_group_3_kernel', 'type': 'discrete', 'domain': range(4, 16)},
              {'name': 'layer_group_3_n_convs', 'type': 'discrete', 'domain': range(8, 32)},
              {'name': 'layer_group_3_max_pool', 'type': 'discrete', 'domain': range(4, 16)},
              {'name': 'layer_group_4_kernel,', 'type': 'discrete', 'domain': range(4, 16)},
              {'name': 'layer_group_4_n_convs', 'type': 'discrete', 'domain': range(8, 512)},
              {'name': 'dense_1_n_hidden', 'type': 'discrete', 'domain': range(64, 1024)},
              {'name': 'dense_2_n_hidden', 'type': 'discrete', 'domain': range(64, 1024)},
              {'name': 'dropout_prob', 'type': 'continuous', 'domain': (0.05, 0.75)},
              {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.000001, 0.01)},
              ]

    mfcc_bounds = [{'name': 'batch_size', 'type': 'discrete', 'domain': range(64, 256)},
                   {'name': 'layer_group_1_kernel', 'type': 'discrete', 'domain': range(4, 8)},
                   {'name': 'layer_group_1_n_convs', 'type': 'discrete', 'domain': range(8, 64)},
                   {'name': 'layer_group_2_kernel', 'type': 'discrete', 'domain': range(8, 32)},
                   {'name': 'layer_group_2_n_convs', 'type': 'discrete', 'domain': range(4, 64)},
                   {'name': 'layer_group_3_kernel', 'type': 'discrete', 'domain': range(4, 16)},
                   {'name': 'layer_group_3_n_convs', 'type': 'discrete', 'domain': range(8, 32)},
                   {'name': 'layer_group_4_kernel,', 'type': 'discrete', 'domain': range(4, 16)},
                   {'name': 'layer_group_4_n_convs', 'type': 'discrete', 'domain': range(8, 512)},
                   {'name': 'dense_1_n_hidden', 'type': 'discrete', 'domain': range(64, 1024)},
                   {'name': 'dropout_prob', 'type': 'continuous', 'domain': (0.05, 0.75)},
                   {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.000001, 0.001)},
                   ]
    myProblem = GPyOpt.methods.BayesianOptimization(gpyopt_helper, mfcc_bounds)
    myProblem.run_optimization(100)
    myProblem.save_evaluations("ev_file")


if __name__ == "__main__":
    # bayes_opt()
    sc = SoundClassifier()
    sc.train_tf_records()