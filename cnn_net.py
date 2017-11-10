# -*-coding:utf-8-*-
import os, sys
import tensorflow as tf
from functools import reduce

from data_process import decode_from_tfrecords, encode_to_tfrecords, get_batch

CONV_LAYER = 'convolutional layer'
FC_LAYER = 'fully-connected layers'
POOL_LAYER = 'pooling layer'
RAW_IMAGE_SHAPE = [-1, 64, 64, 1]
CHAR_CLASS = 21


# shape should be [batch,hight,width,out_channels]

class Layer:
    def __init__(self, type, name):
        self.type = type
        self.name = name
        self.output_shape = None
        self.shape = None

    def get_shape(self, shape):
        raise Exception('not defined inference!')

    def inference(self, inputs):
        raise Exception('not defined inference!')

    @staticmethod
    def _shap_to_list(shape):
        return [i.value for i in shape]


class FullConnectedLayer(Layer):
    # shape should be [m,n]
    def __init__(self, name, x):
        super(FullConnectedLayer, self).__init__(type=FC_LAYER, name=name)
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


class ConvolutionalLayer(Layer):
    def __init__(self, name, x, strides=[1] * 4):
        super(ConvolutionalLayer, self).__init__(type=CONV_LAYER, name=name)
        self.strides = strides
        self.x = x

    def get_shape(self, shape):
        self.shape = [3, 3] + shape[-1:] + [self.x]
        with tf.variable_scope("weights"):
            self.weights = tf.get_variable(name=self.name, shape=self.shape,
                                           initializer=tf.contrib.layers.xavier_initializer_conv2d())
        with tf.variable_scope('biases'):
            self.biases = tf.get_variable(name=self.name, shape=self.shape[-1:],
                                          initializer=tf.contrib.layers.xavier_initializer_conv2d())

    def inference(self, inputs):
        result = tf.nn.relu(tf.nn.conv2d(inputs, self.weights, strides=self.strides, padding='VALID'))
        result = tf.nn.max_pool(result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        self.output_shape = self._shap_to_list(result.shape)
        return result


# class PoolLayer(Layer):
#     def __init__(self, name, strides=[1, 2, 2, 1], shape=[1, 2, 2, 1]):
#         super(PoolLayer, self).__init__(type=POOL_LAYER, name=name, shape=shape)
#         self.strides = strides
#
#     def inference(self, inputs):
#         result = tf.nn.max_pool(inputs, ksize=self.shape, strides=self.strides, padding='VALID')
#         self.output_shape = self._shap_to_list(result.shape)
#         return result


class ConvolutionalNetwork:
    LAYER_DICT = {CONV_LAYER: ConvolutionalLayer, FC_LAYER: FullConnectedLayer, }  # POOL_LAYER: PoolLayer

    def __init__(self):
        self.layers = []

    def addLayer(self, type, name, x):
        # layer_class = self.LAYER_DICT[type]
        if type == FC_LAYER:
            layer = FullConnectedLayer(name, x=x)
        if type == CONV_LAYER:
            layer = ConvolutionalLayer(name, x=x)
        self.layers.append(layer)

    def inference(self, inputs):
        inputs = (tf.cast(inputs, tf.float32) / 255. - 0.5) * 2

        shape = RAW_IMAGE_SHAPE
        for layer in self.layers:
            layer.get_shape(shape=shape)
            inputs = layer.inference(inputs)
            shape = layer.output_shape
        return inputs

    @staticmethod
    def sorfmax_loss(predicts, labels):
        labels = tf.one_hot(labels, depth=CHAR_CLASS, on_value=1.0, off_value=0.0)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predicts))
        return loss

    @staticmethod
    def optimer(loss, lr=0.001):
        return tf.train.AdamOptimizer(lr).minimize(loss)


if __name__ == '__main__':
    net = ConvolutionalNetwork()
    # net.addLayer(FC_LAYER, '1', 100)
    for i in range(3):
        net.addLayer(CONV_LAYER, 'conv' + str(i), 8 ** (i + 1))
    # net.addLayer(FC_LAYER, 'fc0', 1024)
    net.addLayer(FC_LAYER, 'fc1', CHAR_CLASS)

    if not os.path.exists('save'):
        os.mkdir('save')
    if not os.path.exists('save/train.tfrecords'):
        encode_to_tfrecords(record_file='save/train.tfrecords')
    encode_to_tfrecords(record_file='save/text.tfrecords')
    encode_to_tfrecords(record_file='save/train.tfrecords')
    images, labels = decode_from_tfrecords(filename='save/train.tfrecords')
    batch_image, batch_label = get_batch(images, labels, batch_size=30, istrain=True)  # batch 生成测试

    inf = net.inference(batch_image)
    loss = net.sorfmax_loss(inf, batch_label)
    opti = net.optimer(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(inf, axis=1), tf.int32, name='prediction'), batch_label)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        with sys.stdout as file:
            for i in range(10000):  # train 350
                loss_np, _, label_np, image_np, inf_np = session.run([loss, opti,
                                                                      batch_label, batch_image, inf])
                if i % 100 == 0:
                    print('step : %d   trainloss:' % (i), loss_np, file=file)
                    print('train accuracy:', session.run([accuracy]), file=file)
                    # print('text accuracy:', session.run([text_accuracy]))
        coord.request_stop()  # queue需要关闭，否则报错
        coord.join(threads)
