#!/usr/bin/env python
# -*- coding: utf-8 -*-


from model import inference
from model import IMAGE_SIZE

import cv2
import numpy as np
import tensorflow as tf


def predict():
    with tf.Graph().as_default():
        session_config = tf.ConfigProto()
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.4
        with tf.Session(config=session_config) as sess:

            images_holder = tf.placeholder(tf.float32, [1, IMAGE_SIZE, IMAGE_SIZE, 3])
            logits = inference(images_holder, 5, is_training=True)

            init_op = tf.global_variables_initializer()
            sess.run([init_op])
            #restorer = tf.train.Saver()
            #restorer.restore(sess, "logs/model.ckpt-2025006")
            images = np.ndarray([1, IMAGE_SIZE, IMAGE_SIZE, 3])

            im = cv2.imread("test.jpg")
            im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
            #im = np.asarray(im, np.uint8)
            #im = np.asarray(im, np.float32)
            images[0] = im
            #cv2.imshow("image", im)
            #cv2.waitKey(0)

            predicts = sess.run([logits], feed_dict={images_holder: images})
            print predicts
            print np.asarray(predicts).shape

            sess.close()

if __name__ == "__main__":
    predict()