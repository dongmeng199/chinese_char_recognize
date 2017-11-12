# -*-coding:utf-8-*-
import json
import os, sys
import pickle

import tensorflow as tf
from functools import reduce

from data_process import decode_from_tfrecords, encode_to_tfrecords, get_batch

CONV_LAYER = 'convolutional layer'
FC_LAYER = 'fully-connected layers'
POOL_LAYER = 'pooling layer'
DROP_LAYER = 'dropout layer'
RAW_IMAGE_SHAPE = [-1, 64, 64, 1]
CHAR_CLASS = 100
LEARN_RATE = 0.0001
BATCH_SIZE = 100


class Layer:
    def __init__(self, name):
        self.name = name
        self.output_shape = None
        self.shape = None

    def __repr__(self):
        return "name:%s\noutshape:%s\nweights:%s\n\n" % (self.name, self.output_shape, self.weights)

    def get_shape(self, shape):
        raise Exception('not defined inference!')

    def inference(self, inputs):
        raise Exception('not defined inference!')

    @staticmethod
    def _shap_to_list(shape):
        return [i.value for i in shape]

    def obj_to_dict(self, sess):
        raise Exception('not defined inference!')

    def reload_from_dict(self, dict):
        raise Exception('not defined inference!')


class FullConnectedLayer(Layer):
    # shape should be [m,n]
    def __init__(self, name, x):
        super(FullConnectedLayer, self).__init__(name=name)
        self.biases = None
        self.weights = None
        self.x = x

    def get_shape(self, shape):
        self.shape = [reduce(lambda x, y: x * y, shape[1:])] + [self.x]
        with tf.variable_scope("weights"):
            self.weights = tf.get_variable(name=self.name, shape=self.shape,
                                           initializer=tf.contrib.layers.xavier_initializer_conv2d())
        with tf.variable_scope('biases'):
            self.biases = tf.get_variable(name=self.name, shape=self.shape[-1:],
                                          initializer=tf.contrib.layers.xavier_initializer_conv2d())

    def inference(self, inputs):
        inputs = tf.reshape(inputs, [-1, self.shape[0]])
        output = tf.nn.relu(tf.matmul(inputs, self.weights) + self.biases)
        self.output_shape = self._shap_to_list(output.shape)
        return output

    def obj_to_dict(self, sess):
        return {'type': self.__class__, 'weights': sess.run(self.weights), 'biases': sess.run(self.biases),
                'shape': self.shape, 'name': self.name}

    def reload_from_dict(self, dict):
        self.weights = tf.Variable(dict['weights'])
        self.biases = tf.Variable(dict['biases'])
        self.shape = dict['shape']


class ConvolutionalLayer(Layer):
    def __init__(self, name, x, strides=[1] * 4):
        super(ConvolutionalLayer, self).__init__(name=name)
        self.biases = None
        self.weights = None
        self.strides = strides
        self.x = x

    def get_shape(self, shape):
        self.shape = [2, 2] + shape[-1:] + [self.x]
        with tf.variable_scope("weights"):
            self.weights = tf.get_variable(name=self.name, shape=self.shape,
                                           initializer=tf.contrib.layers.xavier_initializer_conv2d())
        with tf.variable_scope('biases'):
            self.biases = tf.get_variable(name=self.name, shape=self.shape[-1:],
                                          initializer=tf.contrib.layers.xavier_initializer_conv2d())

    def inference(self, inputs):
        result = tf.nn.conv2d(inputs, self.weights, strides=self.strides, padding='VALID') + self.biases
        result = tf.nn.relu(result)
        result = tf.nn.max_pool(result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        self.output_shape = self._shap_to_list(result.shape)
        return result

    def obj_to_dict(self, sess):
        return {'type': self.__class__, 'weights': sess.run(self.weights), 'biases': sess.run(self.biases),
                'strides': self.strides, 'name': self.name}

    def reload_from_dict(self, dict):
        self.weights = tf.Variable(dict['weights'])
        self.biases = tf.Variable(dict['biases'])
        self.strides = dict['strides']


class DropOutLayer(Layer):
    def __init__(self, name, x=0.5):
        self.name = name
        self.x = x

    def inference(self, inputs):
        return tf.nn.dropout(inputs, self.x)

    def get_shape(self, shape):
        self.output_shape = shape
        self.weights = tf.get_variable(name=self.name, shape=[],
                                       initializer=tf.contrib.layers.xavier_initializer_conv2d())

    def obj_to_dict(self, sess):
        return {'type': self.__class__, 'x': self.x, 'name': self.name}

    def reload_from_dict(self, dict):
        self.x = dict['x']


class ConvolutionalNetwork:
    LAYER_DICT = {CONV_LAYER: ConvolutionalLayer, FC_LAYER: FullConnectedLayer, }  # POOL_LAYER: PoolLayer

    def __init__(self):
        self.layers = []

    def addLayer(self, type, name, x):
        if type == FC_LAYER:
            layer = FullConnectedLayer(name, x=x)
        if type == CONV_LAYER:
            layer = ConvolutionalLayer(name, x=x)
        if type == DROP_LAYER:
            layer = DropOutLayer(name, x=x)
        self.layers.append(layer)

    def inference(self, inputs, init=False):
        inputs = (tf.cast(inputs, tf.float32) / 255)

        shape = RAW_IMAGE_SHAPE
        for layer in self.layers:
            if init:
                layer.get_shape(shape=shape)
                inputs = layer.inference(inputs)
                shape = layer.output_shape
            else:
                inputs = layer.inference(inputs)
        return inputs

    def obj_to_dict(self, sess):
        return [layer.obj_to_dict(sess) for layer in self.layers]

    def reload_from_dict(self, list):
        for layer_dict in list:
            layer = layer_dict['type'](layer_dict['name'], 1)
            layer.reload_from_dict(layer_dict)
            self.layers.append(layer)

    @staticmethod
    def sorfmax_loss(predicts, labels):
        labels = tf.one_hot(labels, depth=CHAR_CLASS, on_value=1.0, off_value=0.0)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predicts))
        return loss

    @staticmethod
    def optimer(loss, lr=LEARN_RATE):
        return tf.train.AdamOptimizer(lr).minimize(loss)


def train(times=3000, continue_train=False, learn_rate=LEARN_RATE):
    global LEARN_RATE
    LEARN_RATE = learn_rate
    # 加载模型
    net = ConvolutionalNetwork()
    if continue_train:
        net.reload_from_dict(pickle.load(open('save/save1', 'rb')))
    else:
        for i in range(2):
            net.addLayer(CONV_LAYER, 'conv' + str(i), 8 ** (i + 1))
        net.addLayer(FC_LAYER, 'fc1', 1024)
        net.addLayer(DROP_LAYER, 'drop0', 0.5)
        net.addLayer(FC_LAYER, 'fc2', CHAR_CLASS)
    # 准备数据
    if not os.path.exists('save'):
        os.mkdir('save')
    if not os.path.exists('save/train.tfrecords'):
        encode_to_tfrecords(record_file='save/train.tfrecords')
    if not os.path.exists('save/text.tfrecords'):
        encode_to_tfrecords(record_file='save/text.tfrecords')
    images, labels = decode_from_tfrecords(filename='save/train.tfrecords')
    batch_image, batch_label = get_batch(images, labels, batch_size=BATCH_SIZE)
    text_images, text_labels = decode_from_tfrecords(filename='save/text.tfrecords')
    text_batch_image, text_batch_label = get_batch(text_images, text_labels, batch_size=50)
    # 训练用节点
    inf = net.inference(batch_image, init=not continue_train)
    loss = net.sorfmax_loss(inf, batch_label)
    opti = net.optimer(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(inf, axis=1), tf.int32, name='prediction'), batch_label)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 验证用节点
    text_inf = net.inference(text_batch_image, init=False)
    text_correct_prediction = tf.equal(tf.cast(tf.argmax(text_inf, axis=1), tf.int32, name='prediction'),
                                       text_batch_label)
    text_accuracy = tf.reduce_mean(tf.cast(text_correct_prediction, tf.float32))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        with sys.stdout as file:
            for i in range(times):  # train 350
                loss_np, _, label_np, image_np, inf_np = session.run([loss, opti,
                                                                      batch_label, batch_image, inf])
                if i % 100 == 0:
                    print('step : %d   trainloss:' % (i), loss_np, file=file)
                    print('train accuracy:', session.run([accuracy]), file=file)
                    print('text accuracy:', session.run([text_accuracy]), file=file)
        # 存储模型
        pickle.dump(net.obj_to_dict(session), open('save/save2', 'wb'))

        coord.request_stop()  # queue需要关闭，否则报错
        coord.join(threads)


def text(times=300):
    net = ConvolutionalNetwork()
    net.reload_from_dict(pickle.load(open('save/save2', 'rb')))
    # 准备数据
    if not os.path.exists('save'):
        os.mkdir('save')
    if not os.path.exists('save/train.tfrecords'):
        encode_to_tfrecords(record_file='save/train.tfrecords')
    if not os.path.exists('save/text.tfrecords'):
        encode_to_tfrecords(record_file='save/text.tfrecords')
    images, labels = decode_from_tfrecords(filename='save/text.tfrecords')
    batch_image, batch_label = get_batch(images, labels, batch_size=BATCH_SIZE)
    inf = net.inference(batch_image, init=False)
    correct_prediction = tf.equal(tf.cast(tf.argmax(inf, axis=1), tf.int32, name='prediction'), batch_label)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        accuracy_total = 0
        with sys.stdout as file:
            for i in range(times):  # train 350
                accuracy_np = session.run([accuracy])[0]
                print(accuracy_np)
                accuracy_total += accuracy_np
            print(accuracy_total / times)
        coord.request_stop()  # queue需要关闭，否则报错
        coord.join(threads)


if __name__ == '__main__':
    # train(times=10000,continue_train=False)
    # train(times=10000, continue_train=True, learn_rate=LEARN_RATE / 10000)
    text()
