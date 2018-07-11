import librosa
import numpy as np
import pandas as pd

import tensorflow as tf

def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data - min_data) / (max_data - min_data + 1e-6)
    return data - 0.5


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_records(data_dir, id_list, labels, use_mfcc=True, n_mfcc=80):
    sampling_rate = 24000
    n_files = len(id_list)
    split = int(n_files * .8)
    input_length = sampling_rate * 2
    ids_train = id_list[:split]
    labels_train = labels[:split]
    ids_test = id_list[split:]
    labels_test = labels[split:]
    for mode in ["train", "eval"]:
        if mode == "train":
            id_list = ids_train
            labels = labels_train
        else:
            id_list = ids_test
            labels = labels_test
        tfrecords_filename = 'audio_{}.tfrecords'.format(mode)
        writer = tf.python_io.TFRecordWriter(tfrecords_filename)
        for i, ID in enumerate(id_list):
            if i % 100 == 0:
                print "{} {}".format(i, ID)
            file_path = data_dir + ID

            # Read and Resample the audio
            data, _ = librosa.core.load(file_path, sr=sampling_rate,
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
            if use_mfcc:
                data = librosa.feature.mfcc(data, sr=sampling_rate,
                                            n_mfcc=n_mfcc)
                data = np.expand_dims(data, axis=-1)
            else:
                data = audio_norm(data)[:, np.newaxis]
            label = labels[i]
            label = _int64_feature(label)
            example = tf.train.Example(features=tf.train.Features(feature={
                'n_mfcc': _int64_feature(n_mfcc),
                'features': _bytes_feature(data.tostring()),
                'label': label}))

            writer.write(example.SerializeToString())

        writer.close()


train = pd.read_csv("./train.csv")
test = pd.read_csv("./sample_submission.csv")
LABELS = list(train.label.unique())
label_idx = {label: i for i, label in enumerate(LABELS)}
train.set_index("fname", inplace=True)
test.set_index("fname", inplace=True)
train["label_idx"] = train.label.apply(lambda x: label_idx[x])

fnames = train.index
labels = train.label_idx
print labels[0]


create_records("./audio_train/", fnames, labels)




