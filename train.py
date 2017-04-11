#!/usr/bin/env python
# -*- coding: utf-8 -*-


from model import *
from settings import  *

from pascal_tf_records import read_from_tfrecord

train_log_dir = "logs/"
if not tf.gfile.Exists(train_log_dir):
  tf.gfile.MakeDirs(train_log_dir)

def train():
    with tf.Graph().as_default():
        session_config = tf.ConfigProto()
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.2
        images, labels = read_from_tfrecord(["data/train.tfrecord"], BATCH_SIZE)
        logits, _ = inference(images, NUM_CLASSES)

        #v1.0
        loss = squeeze_loss(logits, labels)

        tf.summary.scalar('losses/total_loss', loss)

        # Specify the optimization scheme:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)

        # create_train_op that ensures that when we evaluate it to get the loss,
        # the update_ops are done and the gradient updates are computed.
        train_tensor = slim.learning.create_train_op(loss, optimizer)

        model_path = tf.train.latest_checkpoint(train_log_dir)

        if model_path:
            variables_to_restore = tf.contrib.slim.get_variables_to_restore()
            init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
                model_path, variables_to_restore)
        else:
            init_fn = None

        # Actually runs training.
        slim.learning.train(train_tensor,
                            train_log_dir,
                            save_summaries_secs=60,
                            save_interval_secs=600,
                            session_config=session_config,
                            init_fn=init_fn)


if __name__ == "__main__":
    train()