import math
import random

import tensorflow as tf
import tensorflow.keras as keras

__all__ = ['PreResNet', 'preresnet18', 'preresnet34', 'preresnet50', 'preresnet101',
           'preresnet152', 'preresnet200']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return keras.layers.Conv2D(out_planes, kernel_size=(3, 3), strides=(stride, stride),
                               padding='same', use_bias=False)


class BasicBlock(keras.models.Model):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = keras.layers.BatchNormalization()
        self.downsample = downsample
        self.stride = stride

        global blockID
        self.blockID = blockID
        blockID += 1
        self.downsampling_ratio = 1.

    def call(self, x):
        residual = x

        out = self.bn1(x)
        out_ = self.relu(out)
        out = self.conv1(out_)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(out_)

        out += residual

        if self.downsampling_ratio < 1:
            # torch.nn.functional.adaptive_avg_pool2d(input, output_size)
            #out = F.adaptive_avg_pool2d(out, int(round(out.size(2)*self.downsampling_ratio)))
            ksize = out.shape[1] // int(round(out.shape[1] * self.downsampling_ratio))
            out = tf.nn.avg_pool2d(out, ksize=ksize, strides=ksize, padding='VALID')

        return out


class Bottleneck(keras.models.Model):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = keras.layers.Conv2D(planes, kernel_size=(1, 1), strides=(1, 1),
                                         padding='same', use_bias=False)
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(planes, kernel_size=(3, 3), strides=(stride, stride),
                                         padding='same', use_bias=False)
        self.bn2 = keras.layers.BatchNormalization()
        self.conv3 =keras.layers.Conv2D(planes * 4, kernel_size=(1, 1),
                                        strides=(1, 1), padding='same', use_bias=False)
        self.bn3 = keras.layers.BatchNormalization() 
        self.relu = keras.layers.ReLU()
        self.downsample = downsample
        self.stride = stride

        global blockID
        self.blockID = blockID
        blockID += 1
        self.downsampling_ratio = 1.

    def call(self, x):
        residual = x

        out = self.bn1(x)
        out_ = self.relu(out)
        out = self.conv1(out_)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(out_)

        out += residual

        if self.downsampling_ratio < 1:
            #out = F.adaptive_avg_pool2d(out, int(round(out.size(2)*self.downsampling_ratio)))
            ksize = out.shape[1] // int(round(out.shape[1] * self.downsampling_ratio))
            out = tf.nn.avg_pool2d(out, ksize=ksize, strides=ksize, padding='VALID')

        return out


class PreResNet(keras.models.Model):

    def __init__(self, block, layers, num_classes=1001):
        self.inplanes = 64
        super(PreResNet, self).__init__()

        global blockID
        blockID = 0

        self.conv1 = keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2),
                                         padding='same', use_bias=False)
        self.bn1 = keras.layers.BatchNormalization() 
        self.relu = keras.layers.ReLU()
        self.maxpool =keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = keras.layers.BatchNormalization()
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = keras.layers.Dense(num_classes)

        self.blockID = blockID
        self.downsampling_ratio = 1.
        self.size_after_maxpool = None

        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        '''

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = keras.Sequential([
                keras.layers.Conv2D(planes * block.expansion, kernel_size=(1, 1),
                                    strides=(stride, stride), padding='same', use_bias=False),
                keras.layers.BatchNormalization()
            ])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return keras.Sequential(layers)

    def stochastic_downsampling(self, blockID, ratio):
        block_chosen = blockID is None and random.randint(-1, self.blockID) or blockID
        downsampling_ratios = ratio is None and [0.5, 0.75] or [ratio, ratio]
        if self.blockID == block_chosen:
            self.downsampling_ratio = downsampling_ratios[random.randint(0,1)]
        else:
            self.downsampling_ratio = 1.
        
        # TEST
        #for m in self.modules():
        for m in self.submodules:
            if isinstance(m, Bottleneck):
                if m.blockID == block_chosen:
                    m.downsampling_ratio = downsampling_ratios[random.randint(0,1)]
                else:
                    m.downsampling_ratio = 1.
    
    #def build(self, input_shape):
        

    def call(self, x, blockID=None, ratio=None):
        #self.stochastic_downsampling(blockID, ratio)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.downsampling_ratio < 1:
            if self.size_after_maxpool is None:
                self.size_after_maxpool = self.maxpool(x).shape[2]
                tf.print("1")
            #x = F.adaptive_max_pool2d(x, int(round(self.size_after_maxpool*self.downsampling_ratio)))
            ksize = x.shape[1] // int(round(self.size_after_maxpool*self.downsampling_ratio)) 
            x = tf.nn.max_pool2d(x, ksize=ksize, strides=ksize, padding='VALID')
            tf.print("2")
        else:
            x = self.maxpool(x)
            tf.print("3")

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.relu(x)
        #x = self.avgpool(x)
        x = tf.nn.avg_pool2d(x, ksize=x.shape[1], strides=x.shape[1], padding="VALID")
        x = tf.reshape(x, [x.shape[0], -1])  # x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def preresnet18(pretrained=False, **kwargs):
    """Constructs a PreResNet-18 model.
    """
    model = PreResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def preresnet34(pretrained=False, **kwargs):
    """Constructs a PreResNet-34 model.
    """
    model = PreResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def preresnet50(pretrained=False, **kwargs):
    """Constructs a PreResNet-50 model.
    """
    model = PreResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def preresnet101(pretrained=False, **kwargs):
    """Constructs a PreResNet-101 model.
    """
    model = PreResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def preresnet152(pretrained=False, **kwargs):
    """Constructs a PreResNet-152 model.
    """
    model = PreResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def preresnet200(pretrained=False, **kwargs):
    """Constructs a PreResNet-200 model.
    """
    model = PreResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model