'''
Reference PyTorch implementation: https://github.com/Tushar-N/blockdrop/blob/master/models/base.py
'''

import tensorflow as tf


class Identity(tf.keras.models.Model):
    def __init__(self):
        super(Identity, self).__init__()
    def call(self, x):
        return x


class Flatten(tf.keras.models.Model):
    def __init__(self):
        super(Flatten, self).__init__()
    def call(self, x):
       return tf.reshape(x, [x.shape[0], -1])

def conv3x3(in_planes, out_planes, stride=1):
    return tf.keras.layers.Conv2D(out_planes, kernel_size=(3, 3), strides=(stride, stride),
                                  padding='same', use_bias=False)

class BasicBlock(tf.keras.models.Model):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, trainable=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = tf.keras.layers.BatchNormalization(trainable=trainable)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = tf.keras.layers.BatchNormalization(trainable=trainable)

    def call(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return out


class Bottleneck(tf.keras.models.Model):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, trainable=True):
        super(Bottleneck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(planes, kernel_size=(1, 1), use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(trainable=trainable)
        self.conv2 = tf.keras.layers.Conv2D(planes, kernel_size=(3, 3), strides=(stride, stride),
                                            padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization(trainable=trainable)
        self.conv3 = tf.keras.layers.Conv2D(planes * 4, kernel_size=(1, 1), use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization(trainable=trainable)
        self.relu = tf.keras.layers.ReLU()
    
    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out


class DownsampleB(tf.keras.models.Model):
    
    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.avg = tf.keras.layers.AveragePooling2D((stride, stride))
        self.expand_ratio = nOut // nIn

    def call(self, x):
        x = self.avg(x)
        return tf.concat([x] + [x * 0]*(self.expand_ratio - 1), 1)
