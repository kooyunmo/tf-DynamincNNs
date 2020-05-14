"""
An original PyTorch implementation is at the following link:
https://github.com/ucbdrive/skipnet/blob/master/imagenet/models.py
"""

import math
import tensorflow as tf
import tensorflow.keras as keras
from threading import Lock


global_lock = Lock()

def conv3x3(in_planes, out_planes, stride=1):
    '''3x3 convolution with padding'''
    return keras.layers.Conv2D(out_planes, kernel_size=(3, 3), stride=(stride, stride),
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

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(keras.models.Model):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = keras.layers.Conv2D(planes, kernel_size=(1, 1), use_bias=False)
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(planes, kernel_size=(3, 3), stride=(stride, stride),
                                         padding='same', use_bias=False)
        self.bn2 = keras.layers.BatchNormalization()
        self.conv3 = keras.layers.Conv2D(planes * 4, kernel_size=(1, 1), use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.downsample = downsample
        self.stride = stride
    
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ======================
# Recurrent Gate  Design
# ======================

def repackage_hidden(h):
    if isinstance(h, tf.Variable):
        return tf.Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


class RNNGate(keras.models.Model):
    """given the fixed input size, return a single layer lstm """
    def __init__(self, input_dim, hidden_dim, rnn_type='lstm'):
        super(RNNGate, self).__init__()
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if self.rnn_type == 'lstm':
            self.rnn = keras.layers.LSTM(hidden_dim)    # nn.LSTM(input_dim, hidden_dim)
        else:
            self.rnn = None
        self.hidden = None

        # reduce dim
        #Conv2d(in_channels=hidden_dim, out_channels=1, kernel_size=1, stride=1)
        self.proj = keras.layers.Conv2D(1, kernel_size=(1, 1), strides=(1, 1))
        self.prob = tf.math.sigmoid     # nn.Sigmoid()

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # return (torch.zeros(1, batch_size, self.hidden_dim).cuda()), torch.zeros(1, batch_size, self.hidden_dim).cuda()))
        return (tf.zeros([1, batch_size, self.hidden_dim]), tf.zeros(1, batch_size, self.hidden_dim))

    def repackage_hidden(self):
        self.hidden = repackage_hidden(self.hidden)

    def forward(self, x):
        batch_size = x.shape[0]  # x.size(0)
        # self.rnn.flatten_parameters()     # skip this

        # out, self.hidden = self.rnn(x.view(1, batch_size, -1), self.hidden)
        out, self.hidden = self.rnn(tf.reshape(x, [1, batch_size, -1]))

        #out = out.squeeze()
        out = tf.squeeze(out)
        #proj = self.proj(out.view(out.size(0), out.size(1), 1, 1,)).squeeze()
        proj = self.proj(tf.reshape(out, [out.shape[0], out.shape[1], 1, 1,])).squeeze()
        prob = self.prob(proj)

        #disc_prob = (prob > 0.5).float().detach() - prob.detach() + prob
        disc_prob = tf.stop_gradient(tf.cast((prob > 0.5), tf.float32)) - tf.stop_gradient(prob) + prob
        #disc_prob = disc_prob.view(batch_size, 1, 1, 1)
        disc_prob = tf.reshape(disc_prob, [batch_size,1,1,1])
        
        return disc_prob, prob


# =======================
# Recurrent Gate Model
# =======================
class RecurrentGatedResNet(keras.models.Model):
    def __init__(self, block, layers, num_classes=1000, embed_dim=10,
                 hidden_dim=10, gate_type='rnn', **kwargs):
        self.inplanes = 64
        super(RecurrentGatedResNet, self).__init__()

        self.num_layers = len(layers)
        self.conv1 = keras.layers.Conv2D(64, kernel_size=(7, 7), stride=(2, 2),
                                         padding='same', use_bias=False)
        self.bn1 = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
        self.maxpool = keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='valid')

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # going to have 4 groups of layers. For the easiness of skipping,
        # We are going to break the sequential of layers into a list of layers.
        self._make_group(block, 64, layers[0], group_id=1, pool_size=56)
        self._make_group(block, 128, layers[1], group_id=2, pool_size=28)
        self._make_group(block, 256, layers[2], group_id=3, pool_size=14)
        self._make_group(block, 512, layers[3], group_id=4, pool_size=7)

        if gate_type == 'rnn':
            self.control = RNNGate(embed_dim, hidden_dim, rnn_type='lstm')
        else:
            print('gate type {} not implemented'.format(gate_type))
            self.control = None

        #self.avgpool = nn.AvgPool2d(7)
        self.avgpool = keras.layers.AveragePooling2D((7, 7))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = keras.layers.Dense(num_classes)

        '''
        # PyTorch
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0) * m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2. / n))
        '''

    def _make_group(self, block, planes, layers, group_id=1, pool_size=56):
        """ Create the whole group """
        for i in range(layers):
            if group_id > 1 and i == 0:
                stride = 2
            else:
                stride = 1

            meta = self._make_layer_v2(block, planes, stride=stride,
                                       pool_size=pool_size)

            setattr(self, 'group{}_ds{}'.format(group_id, i), meta[0])
            setattr(self, 'group{}_layer{}'.format(group_id, i), meta[1])
            setattr(self, 'group{}_gate{}'.format(group_id, i), meta[2])

    def _make_layer_v2(self, block, planes, stride=1, pool_size=56):
        """ create one block and optional a gate module """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = keras.Sequential(
                keras.layers.Conv2D(planes * block.expansion, kernel_size=(1, 1),
                                    stride=(stride, stride), use_bias=False),
                keras.layers.BatchNormalization()
            )
        layer = block(self.inplanes, planes, stride, downsample)
        self.inplanes = planes * block.expansion

        # this is for having the same input dimension to rnn gate.
        gate_layer = keras.Sequential(
            keras.layers.AveragePooling2D((pool_size, pool_size)),
            keras.layers.Conv2D(self.embed_dim, kernel_size=(1, 1), stride=(1, 1))
        )
        
        if downsample:
            return downsample, layer, gate_layer
        else:
            return None, layer, gate_layer

    def repackage_hidden(self):
        self.control.hidden = repackage_hidden(self.control.hidden)

    def forward(self, x):
        """mask_values is for the test random gates"""
        # pdb.set_trace()

        batch_size = x.shape[0]     # x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # reinitialize hidden units
        self.control.hidden = self.control.init_hidden(batch_size)

        masks = []
        gprobs = []
        # must pass through the first layer in first group
        x = getattr(self, 'group1_layer0')(x)
        # gate takes the output of the current layer
        gate_feature = getattr(self, 'group1_gate0')(x)
        mask, gprob = self.control(gate_feature)
        gprobs.append(gprob)
        masks.append(mask.squeeze())
        prev = x  # input of next layer

        for g in range(4):
            for i in range(0 + int(g == 0), self.num_layers[g]):
                if getattr(self, 'group{}_ds{}'.format(g+1, i)) is not None:
                    prev = getattr(self, 'group{}_ds{}'.format(g+1, i))(prev)
                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x)
                prev = x = mask.expand_as(x)*x + (1-mask).expand_as(prev)*prev
                gate_feature = getattr(self, 'group{}_gate{}'.format(g+1, i))(x)
                mask, gprob = self.control(gate_feature)
                if not (g == 3 and i == (self.num_layers[3]-1)):
                    # not add the last mask to masks
                    gprobs.append(gprob)
                    masks.append(tf.squeeze(mask))    #masks.append(mask.squeeze())

        x = self.avgpool(x)
        x = tf.reshape(x, [x.shape[0], -1])   # x.view(x.size(0), -1)
        x = self.fc(x)

        return x, masks, gprobs, self.control.hidden