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
import pickle
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.utils import Sequence, to_categorical
import tensorflow as tf

np.random.seed(1001)


class Config(object):
    def __init__(self,
                 sampling_rate=16000, audio_duration=2, n_classes=41,
                 use_mfcc=False, n_folds=10, learning_rate=0.0001,
                 max_epochs=50, n_mfcc=20):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length / 512)), 1)
        else:
            self.dim = (self.audio_length, 1)


# Mean average percision code from:
# https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
# thanks Wendy Kan!
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


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

    def get_2d_conv_model_default(self, input_tensor=None, use_tensor=False, compile_model=False):
        nclass = 41
        if not use_tensor:
            inp = keras.layers.Input(shape=(40, 1 + int(np.floor(44100 * 2 / 512)), 1))
        else:
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
            opt = optimizers.Adam(self.learning_rate)
            model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        return model

    def get_2d_conv_model(self, input_tensor=None, use_tensor=False, compile_model=False):
        nclass = 41
        if not use_tensor:
            inp = keras.layers.Input(shape=(40, 1 + int(np.floor(44100 * 2 / 512)), 1))
        else:
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
            opt = optimizers.Adam(self.learning_rate)
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
        features = tf.cast(features, tf.float32)
        features = tf.reshape(features, (40, 1 + int(np.floor(44100 * 2 / 512)), 1))
        return features

    def fn_parser(self, record):
        keys_to_features = {
            "filename": tf.FixedLenFeature([], tf.string)
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        # features = tf.decode_raw(parsed["features"], tf.string)
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
        """Training function which uses tfrecords file contianing MFCCs
        input:
            max_epochs: the number of training epochs to run"""

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
        model_train = self.get_2d_conv_model_default(input_tensor=x_it.get_next(), compile_model=False, use_tensor=True)
        opt = optimizers.Adam(self.learning_rate)
        model_train.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'],
                            target_tensors=[y_it.get_next()])
        test_dataset = tf.data.TFRecordDataset(filenames=["./audio_40_mfcc_norm_eval.tfrecords"]).repeat()
        test_x = test_dataset.map(self.feature_parser)
        x_it = test_x.batch(self.batch_size).make_one_shot_iterator()

        test_y = test_dataset.map(self.label_parser)
        y_it = test_y.batch(self.batch_size).make_one_shot_iterator()

        model_test = self.get_2d_conv_model_default(compile_model=True)
        opt = optimizers.Adam(self.learning_rate)
        # TODO: Shuffle after each epoch
        for i in range(max_epochs):
            print "cycle {}".format(i)
            model_train.fit(steps_per_epoch=train_set_size / self.batch_size, callbacks=callbacks_list)
            # model_train.fit(steps_per_epoch=1, callbacks=callbacks_list)
            model_train.save_weights("model.h5")

            model_test.load_weights("model.h5")

            predictions = []
            for j in range(int(test_set_size / self.batch_size)):
                x = np.array(keras.backend.get_session().run(x_it.get_next()))
                prediction = model_test.predict(x)
                predictions.append(np.argmax(prediction, axis=1))
            predictions = np.array(predictions).flatten()
            ground_truth = train["label_idx"][train_set_size:train_set_size + predictions.shape[0]]
            print "precision {}".format(precision_score(ground_truth, predictions, average="macro"))
            print confusion_matrix(ground_truth, predictions)
            accuracy = accuracy_score(ground_truth, predictions)
            print "validation accuracy: {}".format(accuracy)
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy

    def prepare_data(self, df, config, data_dir, pickle_name):
        if not os.path.exists(pickle_name):
            X = np.empty(shape=(df.shape[0], config.dim[0], config.dim[1], 1))
            input_length = config.audio_length
            for i, fname in enumerate(df.index):
                print(fname)
                file_path = data_dir + fname
                data, _ = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_fast")

                # Random offset / Padding
                if len(data) > input_length:
                    max_offset = len(data) - input_length
                    offset = np.random.randint(max_offset)
                    data = data[offset:(input_length + offset)]
                else:
                    if input_length > len(data):
                        max_offset = input_length - len(data)
                        offset = np.random.randint(max_offset)
                    else:
                        offset = 0
                    data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

                data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
                data = np.expand_dims(data, axis=-1)
                X[i,] = data
            x = {"x": X}
            with open(pickle_name, 'wb') as handle:
                pickle.dump(x, handle)
            return X
        else:
            with open(pickle_name, 'rb') as handle:
                X = pickle.load(handle)["x"]
            return X

    def train(self):
        train = pd.read_csv("./train.csv")
        test = pd.read_csv("./sample_submission.csv")
        LABELS = list(train.label.unique())
        label_idx = {label: i for i, label in enumerate(LABELS)}
        train.set_index("fname", inplace=True)
        test.set_index("fname", inplace=True)
        train["label_idx"] = train.label.apply(lambda x: label_idx[x])
        config = Config(sampling_rate=44100, audio_duration=2, n_folds=10,
                        learning_rate=0.001, use_mfcc=True, n_mfcc=40)
        X_train = self.prepare_data(train, config, './audio_train/', "mfcc_train.pickle")
        X_test = self.prepare_data(test, config, './audio_test/', "mfcc_test.pickle")
        y_train = to_categorical(train.label_idx, num_classes=config.n_classes)

        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        PREDICTION_FOLDER = "predictions_2d_conv"
        if not os.path.exists(PREDICTION_FOLDER):
            os.mkdir(PREDICTION_FOLDER)
        if os.path.exists('logs/' + PREDICTION_FOLDER):
            shutil.rmtree('logs/' + PREDICTION_FOLDER)

        final_val_split = int(len(train["label_idx"]) * .9)
        final_val = train["label_idx"][final_val_split:]
        train = train[:final_val_split]
        skf = StratifiedKFold(train.label_idx, n_folds=config.n_folds)
        final_predictions = np.ones((len(final_val, config.n_classes)))
        for i, (train_split, val_split) in enumerate(skf):
            keras.backend.clear_session()
            X, y, X_val, y_val = X_train[train_split], y_train[train_split], X_train[val_split], y_train[val_split]
            checkpoint = keras.callbacks.ModelCheckpoint('best_%d.h5' % i, monitor='val_loss', verbose=1,
                                                         save_best_only=True)
            early = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5)
            tb = keras.callbacks.TensorBoard(log_dir='./logs/' + PREDICTION_FOLDER + '/fold_%i' % i, write_graph=True)
            callbacks_list = [checkpoint, early, tb]
            print("#" * 50)
            print("Fold: ", i)
            model = self.get_2d_conv_model(compile_model=True)
            history = model.fit(X, y, validation_data=(X_val, y_val), callbacks=callbacks_list,
                                batch_size=64, epochs=config.max_epochs)
            model.load_weights('best_%d.h5' % i)

            # Save train predictions
            predictions = model.predict(X_train, batch_size=64, verbose=1)
            np.save(PREDICTION_FOLDER + "/train_predictions_%d.npy" % i, predictions)

            # Save test predictions
            predictions = model.predict(X_test, batch_size=64, verbose=1)
            np.save(PREDICTION_FOLDER + "/test_predictions_%d.npy" % i, predictions)

            # Make a submission file
            top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
            predicted_labels = [' '.join(list(x)) for x in top_3]
            test['label'] = predicted_labels
            test[['label']].to_csv(PREDICTION_FOLDER + "/predictions_%d.csv" % i)

            predictions = model.predict(X_val, batch_size=64, verbose=1)
            map3_pred = np.argsort(-predictions, axis=1)[:, :3].reshape((3, -1)).tolist()
            map3_labels = np.argmax(y_val, axis=1).flatten().reshape(-1, 1).tolist()
            map3 = mapk(map3_labels, map3_pred)
            print "MAP3: {}".format(map3)
            if map3 > self.best_accuracy:
                self.best_accuracy = map3

            predictions = model.predict(X_train[final_val_split:], batch_size=64, verbose=1)
            final_predictions = np.multiply(final_predictions, predictions)
            map3_pred = np.argsort(-predictions, axis=1)[:, :3].reshape((3, -1)).tolist()
            map3_labels = final_val.reshape(-1, 1).tolist()
            map3 = mapk(map3_labels, map3_pred)
            print "Final Val MAP3: {}".format(map3)
            if map3 > self.best_accuracy:
                self.best_accuracy = map3

        final_predictions = final_predictions ** (1/config.n_folds)
        map3_pred = np.argsort(-predictions, axis=1)[:, :3].reshape((3, -1)).tolist()
        map3_labels = final_val.reshape(-1, 1).tolist()
        map3 = mapk(map3_labels, map3_pred)
        print "Final Val All Folds MAP3: {}".format(map3)
        if map3 > self.best_accuracy:
            self.best_accuracy = map3



def gpyopt_helper(x):
    """Objective function for GPyOpt.
    args:
        x: a 2D numpy array containing hyperparameters for the current acquisition
    returns:
        Error: The best test error for the training run."""

    sc = SoundClassifier()
    sc.set_params(x[0])
    # sc.train_tf_records()
    sc.train()
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
    bayes_opt()
    # sc = SoundClassifier()
    # sc.train()
