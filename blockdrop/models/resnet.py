import tensorflow as tf
import numpy as np

from models import base

class FlatResNet(tf.keras.models.Model):

    def seed(self, x):
        #x = self.relu(self.bn1(self.conv1(x)))      # CIFAR
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))        # ImageNet
        #raise NotImplementedError

    # This is different from PyTorch nn.Module forward function.
    # run a variable policy batch through the resnet implemented as a full mask over the residual
    # fast to train, non-indicative of time saving (use forward_single instead)
    def forward(self, x, policy):

        x = self.seed(x)

        t = 0
        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                action = policy[: ,t]
                residual = self.ds[segment](x) if b==0 else x

                if tf.reduce_sum(action) == 0:
                    x = residual
                    t += 1
                    continue
                
                action_mask = tf.reshape(tf.cast(action, tf.float32), [-1,1,1,1])
                fx = tf.nn.relu(x)


class FlatResNet32(FlatResNet):
    
    def __init__(self, block, layers, num_classes=10):
        super(FlatResNet32, self).__init__()

        self.inplanes = 16
        self.conv1 = base.conv3x3(3, 16)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.avgpool = tf.keras.layers.AveragePooling2D((8, 8))

        strides = [1, 2, 2]
        filt_sizes = [16, 32, 64]
        self.blocks, self.ds = [], []
        for idx,(filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(blocks)
            self.ds.append(ds)

        self.fc = tf.keras.layers.Dense(num_classes)
        self.fc_dim = 64 * block.expansion

        self.layer_config = layers

        '''
        for m in self.submodules:
            if isinstance(m, tf.keras.layers.Conv2D):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        '''

    def seed(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return x
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = tf.keras.Sequential()
        if stride != 1 or self.inplanes != planes * blocks.expansion:
            downsample = base.DownsampleB(self.inplanes, planes * block.expansion, stride)

        layers = []
        
