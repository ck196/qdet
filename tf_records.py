#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# image supposed to have shape: 480 x 640 x 3 = 921600
IMAGE_PATH = 'data/'


def get_image_binary(filename):
    """ You can read in the image using tensorflow too, but it's a drag
        since you have to create graphs. It's much easier using Pillow and NumPy
    """
    image = cv2.imread(filename)
    image = cv2.resize(image, (224, 224))
    image = np.asarray(image, np.uint8)
    return image.tobytes()  # convert image to raw data bytes in the array.


def write_to_tfrecord(writer, label, binary_image, tfrecord_file):
    """ This example is to write a sample to TFRecord file. If you want to write
    more samples, just use a loop.
    """
    # writer = tf.python_io.TFRecordWriter(tfrecord_file)
    # write image content to the TFRecord file
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_image]))
    }))
    writer.write(example.SerializeToString())


def write_tfrecord(writer, label, image_file, tfrecord_file):
    binary_image = get_image_binary(image_file)
    write_to_tfrecord(writer, label, binary_image, tfrecord_file)


def read_from_tfrecord(filenames, batch_size):
    tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)

    # label and image are stored as bytes but could be stored as
    # int64 or float64 values in a serialized tf.Example protobuf.
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                                                features={
                                                    'label': tf.FixedLenFeature([], tf.int64),
                                                    'image': tf.FixedLenFeature([], tf.string),
                                                }, name='features')
    # image was saved as uint8, so we have to decode as uint8.
    image = tf.decode_raw(tfrecord_features['image'], tf.uint8)
    # the image tensor is flattened out, so we have to reconstruct the shape
    image = tf.reshape(image, [224, 224, 3])
    image = tf.to_float(image)
    label = tf.cast(tfrecord_features['label'], tf.int32)
    one_hot = tf.one_hot(label, 5)
    images, labels = tf.train.shuffle_batch([image, one_hot],
                                            batch_size=batch_size,
                                            capacity=30,
                                            num_threads=1,
                                            min_after_dequeue=10)
    return images, labels


def read_tfrecord(tfrecord_file):
    images, labels = read_from_tfrecord([tfrecord_file], 1)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for x in range(1000):
            image, label = sess.run([images, labels])
            im = np.asarray(image[0], np.uint8)
            cv2.imshow("test", im)
            cv2.waitKey(0)

        coord.request_stop()
        coord.join(threads)


def main():
    # assume the image has the label Chihuahua.
    # in practice, you'd want to use binary numbers for your labels to save space

    tfrecord_file = IMAGE_PATH + 'train.tfrecord'
    # writer = tf.python_io.TFRecordWriter(tfrecord_file)

    # with open("data/train.csv", "r") as reader:
    #     for line in reader.readlines():
    #         line = line.replace("\n","")
    #         image_file, label = line.split(",")
    #         label_int = int(label)

    #         write_tfrecord(writer, label_int, image_file, tfrecord_file)

    # writer.close()

    read_tfrecord(tfrecord_file)




    # if __name__ == '__main__':
    #     main()