#!/usr/bin/env python
# -*- coding: utf-8 -*-


from model import inference
from model import IMAGE_SIZE

import cv2
import numpy as np
import tensorflow as tf

from settings import *

def predict():
    with tf.Graph().as_default():
        session_config = tf.ConfigProto()
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.4
        with tf.Session(config=session_config) as sess:

            images_holder = tf.placeholder(tf.float32, [1, IMAGE_SIZE, IMAGE_SIZE, 3])
            logits, _ = inference(images_holder, 21, is_training=False)

            init_op = tf.global_variables_initializer()
            sess.run([init_op])
            restorer = tf.train.Saver()
            restorer.restore(sess, "logs/model.ckpt-68830")
            images = np.ndarray([1, IMAGE_SIZE, IMAGE_SIZE, 3])

            im = cv2.imread("data/2007_001185.jpg")
            im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
            im_copy = np.asarray(im.copy(), np.float32)
            images[0] = im_copy
            #cv2.imshow("image", im)
            #cv2.waitKey(0)

            predicts = sess.run([logits], feed_dict={images_holder: images})
            results = np.reshape(predicts, (-1, CELL, CELL, BOX_PER_CELL, NUM_CLASSES + 4))

            for i in range(CELL):
                for j in range(CELL):
                    for k in range(BOX_PER_CELL):
                        label = results[0][i][j][k][:-4]
                        cls = np.argmax(label)

                        box = results[0][i][j][k][-4:]
                        if label.any():
                            #print cls, label
                            #print box
                            #box = box
                            cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)

            sess.close()

if __name__ == "__main__":
    predict()