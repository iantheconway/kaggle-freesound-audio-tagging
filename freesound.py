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


class DataGenerator(keras.utils.Sequence):
    def __init__(self, config, data_dir, list_IDs, labels=None,
                 batch_size=64, preprocessing_fn=lambda x: x):
        self.config = config
        self.data_dir = data_dir
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.preprocessing_fn = preprocessing_fn
        self.on_epoch_end()
        self.dim = self.config.dim

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        return self.__data_generation(list_IDs_temp)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        cur_batch_size = len(list_IDs_temp)
        # unpack dimension iterable and put it in a list, for python 2.7
        dims = [cur_batch_size]
        [dims.append(i) for i in self.dim]
        X = np.empty(dims)

        input_length = self.config.audio_length
        for i, ID in enumerate(list_IDs_temp):
            file_path = self.data_dir + ID

            # Read and Resample the audio
            data, _ = librosa.core.load(file_path, sr=self.config.sampling_rate,
                                        res_type='kaiser_fast')

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

            # Normalization + Other Preprocessing
            if self.config.use_mfcc:
                data = librosa.feature.mfcc(data, sr=self.config.sampling_rate,
                                            n_mfcc=self.config.n_mfcc)
                data = np.expand_dims(data, axis=-1)
            else:
                data = self.preprocessing_fn(data)[:, np.newaxis]
            X[i,] = data

        if self.labels is not None:
            y = np.empty(cur_batch_size, dtype=int)
            for i, ID in enumerate(list_IDs_temp):
                y[i] = self.labels[ID]
            return X, keras.utils.to_categorical(y, num_classes=self.config.n_classes)
        else:
            return X


class SoundClassifier(object):
    def __init__(self):
        self.batch_size = 128
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

    def audio_norm(self, data):
        max_data = np.max(data)
        min_data = np.min(data)
        data = (data - min_data) / (max_data - min_data + 1e-6)
        return data - 0.5

    def get_1d_conv_model(self, config):
        nclass = config.n_classes
        input_length = config.audio_length

        inp = keras.layers.Input(shape=(input_length, 1))
        x = keras.layers.Convolution1D(self.layer_group_1_n_convs, self.layer_group_1_kernel, activation=relu,
                                       padding="valid")(
            inp)
        x = keras.layers.Convolution1D(self.layer_group_1_n_convs, self.layer_group_1_kernel, activation=relu,
                                       padding="valid")(x)
        x = keras.layers.MaxPool1D(self.layer_group_1_max_pool)(x)
        x = keras.layers.Dropout(rate=self.dropout_prob)(x)

        x = keras.layers.Convolution1D(self.layer_group_2_n_convs, self.layer_group_2_kernel, activation=relu,
                                       padding="valid")(x)
        x = keras.layers.Convolution1D(self.layer_group_2_n_convs, self.layer_group_2_kernel, activation=relu,
                                       padding="valid")(x)
        x = keras.layers.MaxPool1D(self.layer_group_2_max_pool)(x)
        x = keras.layers.Dropout(rate=self.dropout_prob)(x)

        x = keras.layers.Convolution1D(self.layer_group_3_n_convs, self.layer_group_3_kernel, activation=relu,
                                       padding="valid")(x)
        x = keras.layers.Convolution1D(self.layer_group_3_n_convs, self.layer_group_3_kernel, activation=relu,
                                       padding="valid")(x)
        x = keras.layers.MaxPool1D(self.layer_group_3_max_pool)(x)
        x = keras.layers.Dropout(rate=self.dropout_prob)(x)

        x = keras.layers.Convolution1D(self.layer_group_4_n_convs, self.layer_group_4_kernel, activation=relu,
                                       padding="valid")(x)
        x = keras.layers.Convolution1D(self.layer_group_4_n_convs, self.layer_group_4_kernel, activation=relu,
                                       padding="valid")(x)
        x = keras.layers.GlobalMaxPool1D()(x)
        x = keras.layers.Dropout(rate=0.2)(x)

        x = keras.layers.Dense(self.dense_1_n_hidden, activation=relu)(x)
        x = keras.layers.Dense(self.dense_2_n_hidden, activation=relu)(x)
        out = keras.layers.Dense(nclass, activation=softmax)(x)

        model = models.Model(inputs=inp, outputs=out)
        opt = optimizers.Adam(config.learning_rate)

        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        return model

    def get_2d_conv_model(self, config=None, input_tensor=None, compile_model=False):
        if config:
            nclass = config.n_classes
            print config.dim
            inp = keras.layers.Input(shape=(config.dim[0], config.dim[1], 1))
        else:
            nclass = 41
            inp = keras.layers.Input(tensor=input_tensor)
            print inp
        x = keras.layers.Convolution2D(self.layer_group_1_n_convs,
                                       (self.layer_group_1_kernel, self.layer_group_1_kernel), padding="same")(inp)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D()(x)

        x = keras.layers.Convolution2D(self.layer_group_2_n_convs,
                                       (self.layer_group_2_kernel, self.layer_group_2_kernel), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D()(x)
        x = keras.layers.Convolution2D(self.layer_group_3_n_convs,
                                       (self.layer_group_3_kernel, self.layer_group_3_kernel), padding="same")(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D()(x)
        x = keras.layers.Convolution2D(self.layer_group_4_n_convs,
                                       (self.layer_group_4_kernel, self.layer_group_4_kernel), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D()(x)

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

    def train(self, max_epochs=50, n_folds=10):
        train = pd.read_csv("./train.csv")
        test = pd.read_csv("./sample_submission.csv")
        LABELS = list(train.label.unique())
        label_idx = {label: i for i, label in enumerate(LABELS)}
        train.set_index("fname", inplace=True)
        test.set_index("fname", inplace=True)
        train["label_idx"] = train.label.apply(lambda x: label_idx[x])

        config = Config(sampling_rate=24000, audio_duration=2, n_folds=n_folds, learning_rate=self.learning_rate,
                        max_epochs=max_epochs, use_mfcc=self.use_mfcc, n_mfcc=80)

        PREDICTION_FOLDER = "predictions_1d_conv"
        if not os.path.exists(PREDICTION_FOLDER):
            os.mkdir(PREDICTION_FOLDER)
        if os.path.exists('logs/' + PREDICTION_FOLDER):
            shutil.rmtree('logs/' + PREDICTION_FOLDER)

        skf = StratifiedKFold(train.label_idx, n_folds=config.n_folds)

        for i, (train_split, val_split) in enumerate(skf):
            train_set = train.iloc[train_split]
            val_set = train.iloc[val_split]
            checkpoint = keras.callbacks.ModelCheckpoint('best_%d.h5' % i, monitor='val_loss', verbose=1,
                                                         save_best_only=True)
            early = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5)
            tb = keras.callbacks.TensorBoard(log_dir='./logs/' + PREDICTION_FOLDER + '/fold_%d' % i, write_graph=True)

            callbacks_list = [checkpoint, early, tb]
            print("Fold: ", i)
            print("#" * 50)

            if self.use_mfcc:
                model = self.get_2d_conv_model(config)
            else:
                model = self.get_1d_conv_model(config)
            train_generator = DataGenerator(config, './audio_train/', train_set.index,
                                            train_set.label_idx, batch_size=self.batch_size,
                                            preprocessing_fn=self.audio_norm)
            val_generator = DataGenerator(config, './audio_train/', val_set.index,
                                          val_set.label_idx, batch_size=self.batch_size,
                                          preprocessing_fn=self.audio_norm)
            history = model.fit_generator(train_generator, callbacks=callbacks_list, validation_data=val_generator,
                                          epochs=config.max_epochs, use_multiprocessing=True, workers=6,
                                          max_queue_size=100)

            model.load_weights('best_%d.h5' % i)

            # Save train predictions
            train_generator = DataGenerator(config, './audio_train/', train.index, batch_size=self.batch_size,
                                            preprocessing_fn=self.audio_norm)
            predictions = model.predict_generator(train_generator, use_multiprocessing=True,
                                                  workers=6, max_queue_size=100, verbose=1)
            np.save(PREDICTION_FOLDER + "/train_predictions_%d.npy" % i, predictions)

            # Save test predictions
            # change batch size to 1, otherwise we have missing results.
            test_generator = DataGenerator(config, './audio_test/', test.index, batch_size=1,
                                           preprocessing_fn=self.audio_norm)
            predictions = model.predict_generator(test_generator, use_multiprocessing=True,
                                                  workers=6, max_queue_size=100, verbose=1)
            np.save(PREDICTION_FOLDER + "/test_predictions_%d.npy" % i, predictions)

            # Make a submission file
            top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
            predicted_labels = [' '.join(list(x)) for x in top_3]
            test['label'] = predicted_labels
            test[['label']].to_csv(PREDICTION_FOLDER + "/predictions_%d.csv" % i)

        pred_list = []
        for i in range(config.n_folds):
            pred_list.append(np.load("./predictions_1d_conv/test_predictions_%d.npy" % i))
        prediction = np.ones_like(pred_list[0])
        for pred in pred_list:
            prediction = prediction * pred
        prediction = prediction ** (1. / len(pred_list))
        # Make a submission file
        top_3 = np.array(LABELS)[np.argsort(-prediction, axis=1)[:, :3]]
        predicted_labels = [' '.join(list(x)) for x in top_3]
        test = pd.read_csv('./sample_submission.csv')
        print len(predicted_labels)
        print test.info()
        test['label'] = predicted_labels
        test[['fname', 'label']].to_csv("1d_conv_ensembled_submission.csv", index=False)

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
        features = tf.reshape(features, (80, 1 + int(np.floor(24000 * 2 / 512)), 1))
        return features

    def record_parse(self, record):
        keys_to_features = {
            "features": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.string)
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        features = tf.decode_raw(parsed["features"], tf.uint8)
        features = tf.cast(features, tf.float32)
        features = tf.reshape(features, (80, 1 + int(np.floor(24000 * 2 / 512)), 1))
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
        # train_dataset = tf.data.TFRecordDataset(filenames=["./audio_train.tfrecords"])
        # train_dataset = train_dataset.shuffle(10000)
        # train_dataset = train_dataset.map(self.record_parse).batch(self.batch_size)
        train_dataset = tf.data.TFRecordDataset(filenames=["./audio2_train.tfrecords"])
        train_dataset = train_dataset.shuffle(10000)
        train_x = train_dataset.map(self.feature_parser)
        x_it = train_x.batch(self.batch_size).make_one_shot_iterator()

        train_y = train_dataset.map(self.label_parser)
        y_it = train_y.batch(self.batch_size).make_one_shot_iterator()
        print x_it.get_next().shape
        model = self.get_2d_conv_model(input_tensor=x_it.get_next(), compile_model=False)
        opt = optimizers.Adam(self.learning_rate)
        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'],
                      target_tensors=[y_it.get_next()])
        sess = keras.backend.get_session()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        model.fit(steps_per_epoch=10000/self.batch_size)
        self.best_accuracy = model.evaluate()


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
                   {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.000001, 0.01)},
                   ]
    myProblem = GPyOpt.methods.BayesianOptimization(gpyopt_helper, mfcc_bounds)
    myProblem.run_optimization(100)
    myProblem.save_evaluations("ev_file")


if __name__ == "__main__":
    bayes_opt()
