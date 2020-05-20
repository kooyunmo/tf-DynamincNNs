import tensorflow as tf
import tensorflow.keras as keras

from utils.config import FLAGS

width_mult_list = [0.25, 0.50, 0.75, 1.0]

class SwitchableBatchNorm2d(keras.models.Model):
    def __init__(self, num_features_list):
        super(SwitchableBatchNorm2d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        bns = []
        for i in num_features_list:
            bns.append(keras.layers.BatchNormalization())
        #self.bn = nn.ModuleList(bns)
        self.bn = bns
        self.width_mult = max(width_mult_list)
        self.ignore_model_profiling = True

    def call(self, input):
        idx = width_mult_list.index(self.width_mult)
        y = self.bn[idx](input)
        return y


class SlimmableConv2d(keras.layers.Conv2D):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True):
        super(SlimmableConv2d, self).__init__(
            max(out_channels_list), (kernel_size, kernel_size),
            strides=(stride, stride), padding='same', use_bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult = max(width_mult_list)

    def call(self, input):
        idx = width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        #weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        weight = self.get_weights()[0][:self.out_channels, :self.in_channels, :, :]

        '''
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        '''
        if self.get_weights()[1] is not None:
            bias = self.get_weights()[1][:self.out_channels]
        else:
            bias = self.get_weights()[1]

        #y = nn.functional.conv2d(
        #    input, weight, bias, self.stride, self.padding,
        #    self.dilation, self.groups)
        y = tf.nn.conv2d(input, weight, self.strides, padding='SAME')
        y = tf.nn.bias_add(y, bias)

        return y


class SlimmableLinear(keras.layers.Dense):
    def __init__(self, in_features_list, out_features_list, bias=True):
        super(SlimmableLinear, self).__init__(
            max(out_features_list), use_bias=bias)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.width_mult = max(width_mult_list)

    def call(self, input):
        idx = width_mult_list.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        #weight = self.weight[:self.out_features, :self.in_features]
        weight = self.get_weights()[0][:self.out_features, :self.in_features]
        '''
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        '''
        if self.get_weights()[1] is not None:
            bias = self.get_weights()[1][:self.out_features]
        else:
            bias = self.get_weights()[1]
            
        #return nn.functional.linear(input, weight, bias)
        new_dense = keras.layers.Dense(self.units)
        new_dense(input)        # for weight initialization with real values
        new_dense.set_weights([weight, bias])
        return new_dense(input)


def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class USConv2d(keras.layers.Conv2D):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=True,
                 us=[True, True], ratio=[1, 1]):
        super(USConv2d, self).__init__(
            out_channels, (kernel_size, kernel_size), strides=(stride, stride),
            padding='same', use_bias=bias)
        self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult = None
        self.us = us
        self.ratio = ratio

    def call(self, input):
        if self.us[0]:
            self.in_channels = make_divisible(
                self.in_channels_max
                * self.width_mult
                / self.ratio[0]) * self.ratio[0]
        if self.us[1]:
            self.out_channels = make_divisible(
                self.out_channels_max
                * self.width_mult
                / self.ratio[1]) * self.ratio[1]
        self.groups = self.in_channels if self.depthwise else 1
        #weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        weight = self.get_weights()[0][:self.out_channels, :self.in_channels, :, :]
        #if self.bias is not None:
        #    bias = self.bias[:self.out_channels]
        #else:
        #    bias = self.bias
        if self.get_weights()[1] is not None:
            bias = self.get_weights()[1][:self.out_channels]
        else:
            bias = self.get_weights()[1]
        #y = nn.functional.conv2d(
        #    input, weight, bias, self.stride, self.padding,
        #    self.dilation, self.groups)
        y = tf.nn.conv2d(input, weight, strides=self.strides, padding=self.padding)
        '''
        if getattr(FLAGS, 'conv_averaged', False):
            y = y * (max(self.in_channels_list) / self.in_channels)
        '''
        return y


class USLinear(keras.layers.Dense):
    def __init__(self, in_features, out_features, bias=True, us=[True, True]):
        super(USLinear, self).__init__(out_features, use_bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features
        self.width_mult = None
        self.us = us

    def call(self, input):
        if self.us[0]:
            self.in_features = make_divisible(
                self.in_features_max * self.width_mult)
        if self.us[1]:
            self.out_features = make_divisible(
                self.out_features_max * self.width_mult)
        #weight = self.weight[:self.out_features, :self.in_features]
        weight = self.get_weights()[0][:self.out_features, :self.in_features]
        #if self.bias is not None:
        #    bias = self.bias[:self.out_features]
        #else:
        #    bias = self.bias
        if self.get_weights()[1] is not None:
            bias = self.get_weights()[1][:self.out_features]
        else:
            bias = self.get_weights()[1]
        #return nn.functional.linear(input, weight, bias)
        new_dense = keras.layers.Dense(self.units)
        new_dense(input)        # for weight initialization with real values
        new_dense.set_weights([weight, bias])
        return new_dense(input)


class USBatchNorm2d(keras.layers.BatchNormalization):
    def __init__(self, num_features, ratio=1):
        super(USBatchNorm2d, self).__init__(
            num_features, affine=True, track_running_stats=False)
        self.num_features_max = num_features
        # for tracking performance during training
        #self.bn = nn.ModuleList([
        #    nn.BatchNorm2d(i, affine=False) for i in [
        #        make_divisible(
        #            self.num_features_max * width_mult / ratio) * ratio
        #        for width_mult in width_mult_list]])
        self.bn = [keras.layers.BatchNormalization() for _ in [ 
            make_divisible(self.num_features_max * width_mult / ratio) * ratio
                for width_mult in width_mult_list]]
        self.ratio = ratio
        self.width_mult = None
        self.ignore_model_profiling = True

    def call(self, input):
        weight = self.get_weights()[0]  # self.weight
        bias = self.get_weights()[1]    # self.bias
        c = make_divisible(
            self.num_features_max * self.width_mult / self.ratio) * self.ratio
        '''
        if self.width_mult in width_mult_list:
            idx = width_mult_list.index(self.width_mult)
            y = nn.functional.batch_norm(
                input,
                self.bn[idx].running_mean[:c],
                self.bn[idx].running_var[:c],
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
            y = tf.nn.batch_normalization(input, self.)
        else:
            y = nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        '''
        new_bn = keras.layers.BatchNormalization()
        new_bn.set_weights([weight, bias, self.moving_mean, self.moving_variance])
        y = new_bn(input)
        return y


def pop_channels(autoslim_channels):
    return [i.pop(0) for i in autoslim_channels]


def bn_calibration_init(m):
    """ calculating post-statistics of batch normalization """
    if getattr(m, 'track_running_stats', False):
        # reset all values for post-statistics
        m.reset_running_stats()
        # set bn in training mode to update post-statistics
        m.training = True
        # if use cumulative moving average
        '''
        if getattr(FLAGS, 'cumulative_bn_stats', False):
            m.momentum = None
        '''