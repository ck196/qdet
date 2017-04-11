#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.layers.python.layers import utils

slim = tf.contrib.slim

from settings import  *

losses = slim.losses

@add_arg_scope
def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None,
                outputs_collections=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            outputs_collections=None):
            net = squeeze(inputs, squeeze_depth)
            outputs = expand(net, expand_depth)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)


def squeeze(inputs, num_outputs):
    return slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')


def expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        e1x1 = slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
        e3x3 = slim.conv2d(inputs, num_outputs, [3, 3], scope='3x3')
    return tf.concat([e1x1, e3x3], 3)


def squeezenet_arg_scope(is_training, decay):
    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm],
                            is_training=is_training,
                            decay=decay) as sc:
            return sc


def inference(images,
              num_classes,
              is_training=True,
              batch_norm_decay=0.999):
    with slim.arg_scope(squeezenet_arg_scope(is_training, batch_norm_decay)):
        with tf.variable_scope('squeezenet', values=[images]) as sc:
            end_point_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([fire_module, slim.conv2d,
                                 slim.max_pool2d, slim.avg_pool2d],
                                outputs_collections=[end_point_collection]):
                net = slim.conv2d(images, 96, [7, 7], stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], scope='maxpool1')
                net = fire_module(net, 16, 64, scope='fire2')
                net = fire_module(net, 16, 64, scope='fire3')
                net = fire_module(net, 32, 128, scope='fire4')
                net = slim.max_pool2d(net, [3, 3], scope='maxpool4')
                net = fire_module(net, 32, 128, scope='fire5')
                net = fire_module(net, 48, 192, scope='fire6')
                net = fire_module(net, 48, 192, scope='fire7')
                net = fire_module(net, 64, 256, scope='fire8')
                net = slim.max_pool2d(net, [3, 3], scope='maxpool8')
                net = fire_module(net, 64, 256, scope='fire9')
                net = slim.max_pool2d(net, [1, 1], scope='maxpool9')
                net = fire_module(net, 80, 512, scope='fire10')
                net = slim.fully_connected(net, BOX_PER_CELL * (num_classes + 4) , scope='fully_connected')
                logits = slim.flatten(net, scope="flatten")

                logits = utils.collect_named_outputs(end_point_collection,
                                                     sc.name + '/logits',
                                                     logits)
            end_points = utils.convert_collection_to_dict(end_point_collection)

    return logits, end_points


def squeeze_loss(logits, labels):
    new_logits = tf.reshape(logits, [-1, BATCH_SIZE, CELL, CELL, BOX_PER_CELL, NUM_CLASSES + 4])
    new_lables = tf.reshape(labels, [-1, BATCH_SIZE, CELL, CELL, BOX_PER_CELL, NUM_CLASSES + 4])
    class_predict = new_logits[:, :, :, :, :, :NUM_CLASSES]
    bbox_predict = new_logits[:, :, :, :, :, -4:]

    class_gt = new_lables[:, :, :, :, :, :NUM_CLASSES]
    bbox_gt = new_lables[:, :, :, :, :, -4:]

    bbox_loss = slim.losses.mean_squared_error(bbox_gt, bbox_predict)
    class_loss = slim.losses.softmax_cross_entropy(class_gt, class_predict)
    tf.summary.scalar('losses/bbox_loss', bbox_loss)
    tf.summary.scalar('losses/class_loss', class_loss)
    total_loss = bbox_loss + class_loss
    return total_loss
