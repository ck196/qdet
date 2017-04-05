#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


def t():
    writer = tf.python_io.TFRecordWriter("abc.tfrecords")
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=[1.0, 2.0, 3.0, 3.0, 4.0]))
    }))
    writer.write(example.SerializeToString())
    writer.close()

    tfrecord_file_queue = tf.train.string_input_producer(["abc.tfrecords"], name='queue')
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)

    features = tf.parse_single_example(
        tfrecord_serialized,
        # Defaults are not specified since both keys are required.
        features={
            'label': tf.VarLenFeature(tf.float32)
        })

    label = tf.cast(features['label'].values, tf.float32)
    return label

if __name__ == "__main__":

    label = t()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        label1 = sess.run([label])

        print label1

        coord.request_stop()
        coord.join(threads)