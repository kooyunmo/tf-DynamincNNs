import torch.nn as nn
import tensorflow as tf
import tensorflow.keras as keras
import math


from .slimmable_ops import SwitchableBatchNorm2d
from .slimmable_ops import SlimmableConv2d, SlimmableLinear
#from utils.config import FLAGS

depth = 50
width_mult_list = [0.25, 0.50, 0.75, 1.0]
reset_parameters = True

class Block(keras.models.Model):
    def __init__(self, inp, outp, stride):
        super(Block, self).__init__()
        assert stride in [1, 2]

        midp = [i // 4 for i in outp]
        layers = [
            SlimmableConv2d(inp, midp, 1, 1, 0, bias=False),
            SwitchableBatchNorm2d(midp),
            keras.layers.ReLU(),

            SlimmableConv2d(midp, midp, 3, stride, 1, bias=False),
            SwitchableBatchNorm2d(midp),
            keras.layers.ReLU(),

            SlimmableConv2d(midp, outp, 1, 1, 0, bias=False),
            SwitchableBatchNorm2d(outp),
        ]
        self.body = keras.Sequential(layers)

        self.residual_connection = stride == 1 and inp == outp
        if not self.residual_connection:
            self.shortcut = keras.Sequential([
                SlimmableConv2d(inp, outp, 1, stride=stride, bias=False),
                SwitchableBatchNorm2d(outp),
            ])
        self.post_relu = keras.layers.ReLU()

    def call(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
            res += self.shortcut(x)
        res = self.post_relu(res)
        return res


class Model(keras.models.Model):
    def __init__(self, num_classes=1001, input_size=224):
        super(Model, self).__init__()

        self.features = []
        # head
        assert input_size % 32 == 0

        # setting of inverted residual blocks
        self.block_setting_dict = {
            # : [stage1, stage2, stage3, stage4]
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }
        self.block_setting = self.block_setting_dict[depth]
        feats = [64, 128, 256, 512]
        channels = [
            int(64 * width_mult) for width_mult in width_mult_list]
        self.features.append(
            keras.Sequential([
                SlimmableConv2d(
                    [3 for _ in range(len(channels))], channels, 7, 2, 3,
                    bias=False),
                SwitchableBatchNorm2d(channels),
                keras.layers.ReLU(),
                keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')
            ])
        )

        # body
        for stage_id, n in enumerate(self.block_setting):
            outp = [
                int(feats[stage_id] * width_mult * 4)
                for width_mult in width_mult_list]
            for i in range(n):
                if i == 0 and stage_id != 0:
                    self.features.append(Block(channels, outp, 2))
                else:
                    self.features.append(Block(channels, outp, 1))
                channels = outp

        avg_pool_size = input_size // 32
        self.features.append(keras.layers.AveragePooling2D(pool_size=(avg_pool_size, avg_pool_size)))

        # make it nn.Sequential
        self.features = keras.Sequential(self.features)

        # classifier
        self.outp = channels
        self.classifier = keras.Sequential([
            SlimmableLinear(
                self.outp,
                [num_classes for _ in range(len(self.outp))]
            )
        ])
        '''
        if reset_parameters:
            self.reset_parameters()
        '''

    def call(self, x):
        x = self.features(x)
        last_dim = x.shape[3]
        x = tf.reshape(x, [-1, last_dim])   # x = x.view(-1, last_dim)
        x = self.classifier(x)

        return x

    '''
    def reset_parameters(self):
        for m in self.submodules:
            if isinstance(m, keras.layers.Conv2D):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    '''