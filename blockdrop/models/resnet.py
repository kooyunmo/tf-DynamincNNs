import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from models import base

class FlatResNet(keras.models.Model):

    def seed(self, x):
        #x = self.relu(self.bn1(self.conv1(x)))      # CIFAR
        #x = self.maxpool(self.relu(self.bn1(self.conv1(x))))        # ImageNet
        raise NotImplementedError

    # This is different from PyTorch nn.Module forward function.
    # run a variable policy batch through the resnet implemented as a full mask over the residual
    # fast to train, non-indicative of time saving (use forward_single instead)
    def forward(self, x, policy):
        #print("x1: ", x.shape)
        x = self.seed(x)
        #print("x2: ", x.shape)
        t = 0
        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                action = policy[: ,t]
                residual = self.ds[segment](x) if b==0 else x

                if tf.reduce_sum(action) == 0:
                    x = residual
                    t += 1
                    continue
                
                #print("x3: ", x.shape)
                action_mask = tf.reshape(tf.cast(action, tf.float32), [-1,1,1,1])
                fx = tf.nn.relu(residual + self.blocks[segment][b](x))
                x = fx * action_mask + residual*(1-action_mask)
                t += 1
        x = self.avgpool(x)
        x = tf.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        return x

    # run a single, fixed policy for all items in the batch
    # policy is a (15,) vector. Use with batch_size=1 for profiling
    def forward_single(self, x, policy):
        x = self.seed(x)

        t = 0
        for segment, num_blocks in enumerate(self.layer_config):
           for b in range(num_blocks):
                residual = self.ds[segment](x) if b==0 else x
                if policy[t]==1:
                    x = residual + self.blocks[segment][b](x)
                    x = tf.nn.relu(x)
                else:
                    x = residual
                t += 1

        x = self.avgpool(x)
        x = tf.reshape(x [x.shape[0], -1])
        x = self.fc(x)
        return x


    def forward_full(self, x):
        x = self.seed(x)

        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                residual = self.ds[segment](x) if b==0 else x
                x = tf.nn.relu(residual + self.blocks[segment][b](x))
        x = self.avgpool(x)
        x = tf.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        return x


class FlatResNet32(FlatResNet):
    
    def __init__(self, block, layers, num_classes=10):
        super(FlatResNet32, self).__init__()

        self.inplanes = 16
        self.conv1 = base.conv3x3(3, 16)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.avgpool = tf.keras.layers.AveragePooling2D((8, 8), padding='same')

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


    def seed(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return x
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = tf.keras.Sequential()
        if stride != 1 or self.inplanes != planes * blocks.expansion:
            downsample = base.DownsampleB(self.inplanes, planes * block.expansion, stride)

        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1))

        return layers, downsample


# Regular Flattened Resnet, tailored for Imagenet etc.
class FlatResNet224(FlatResNet):

    def __init__(self, block, layers, num_classes=1001, trainable=True):
        self.inplanes = 64
        super(FlatResNet224, self).__init__()
        self.conv1 = keras.layers.Conv2D(64, kernel_size=7, strides=(2, 2), padding='same', use_bias=False)
        self.bn1 = keras.layers.BatchNormalization(trainable=trainable)
        self.relu = keras.layers.ReLU()
        self.maxpool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')

        strides = [1, 2, 2, 2]
        filt_sizes = [64, 128, 256, 512]
        self.blocks, self.ds = [], []
        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(blocks)
            self.ds.append(ds)

        self.avgpool = keras.layers.AveragePooling2D(pool_size=(7, 7), padding='same')
        self.fc = keras.layers.Dense(num_classes)

        self.layer_config = layers
        self.trainable = trainable

    def seed(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return x

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = keras.Sequential()
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = keras.Sequential([
                keras.layers.Conv2D(planes * block.expansion, kernel_size=1, strides=(stride, stride),
                                    padding='same', use_bias=False),
                keras.layers.BatchNormalization(trainable=self.trainable)
            ])

        layers = [block(self.inplanes, planes, stride, trainable=self.trainable)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, trainable=self.trainable))

        return layers, downsample


#---------------------------------------------------------------------------------------------------------#

# Class to generate resnetNB or any other config (default is 3B)
# removed the fc layer so it serves as a feature extractor
class Policy32(keras.models.Model):

    def __init__(self, layer_config=[1,1,1], num_blocks=15):
        super(Policy32, self).__init__()
        self.features = FlatResNet32(base.BasicBlock, layer_config, num_classes=10)
        #self.feat_dim = self.features.fc.weight.data.shape[1]
        self.features.fc = keras.Sequential()

        self.logit = keras.layers.Dense(num_blocks)
        self.vnet = keras.layers.Dense(1)

    '''
    def load_state_dict(self, state_dict):
        # support legacy models
        state_dict = {k:v for k,v in state_dict.items() if not k.startswith('features.fc')}
        return super(Policy32, self).load_state_dict(state_dict)
    '''

    def call(self, x):
        x = self.features.forward_full(x)
        value = self.vnet(x)
        probs = tf.math.sigmoid(self.logit(x))
        return probs, value


class Policy224(keras.models.Model):

    def __init__(self, layer_config=[1,1,1,1], num_blocks=16, trainable=True):
        super(Policy224, self).__init__()
        self.features = FlatResNet224(base.BasicBlock, layer_config, num_classes=1001, trainable=trainable)

        '''
        resnet18 = torchmodels.resnet18(pretrained=True)
        utils.load_weights_to_flatresnet(resnet18, self.features)
        '''

        self.features.avgpool = keras.layers.AveragePooling2D(pool_size=(4, 4), padding='same')
        #self.feat_dim = self.features.fc.weight.data.shape[1]
        self.features.fc = keras.Sequential()


        self.logit = keras.layers.Dense(num_blocks)
        self.vnet = keras.layers.Dense(1)

    '''
    def load_state_dict(self, state_dict):
        # support legacy models
        state_dict = {k:v for k,v in state_dict.items() if not k.startswith('features.fc')}
        return super(Policy224, self).load_state_dict(state_dict)
    '''

    def call(self, x):
        x = tf.nn.avg_pool2d(input=x, ksize=2, strides=2, padding='SAME')
        x = self.features.forward_full(x)
        value = self.vnet(x)
        probs = tf.math.sigmoid(self.logit(x))
        return probs, value


#--------------------------------------------------------------------------------------------------------#

class StepResnet32(FlatResNet32):

    def __init__(self, block, layers, num_classes, joint=False):
        super(StepResnet32, self).__init__(block, layers, num_classes)
        #self.eval() # default to eval mode

        self.joint = joint

        self.state_ptr = {}
        t = 0
        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                self.state_ptr[t] = (segment, b)
                t += 1

    def seed(self, x):
        self.state = self.relu(self.bn1(self.conv1(x)))
        self.t = 0

        if self.joint:
            return self.state
        return tf.Variable(self.state)

    def step(self, action):
        segment, b = self.state_ptr[self.t]
        residual = self.ds[segment](self.state) if b==0 else self.state
        action_mask = action.float().view(-1,1,1,1)

        fx = tf.nn.relu(residual + self.blocks[segment][b](self.state))
        self.state = fx*action_mask + residual*(1-action_mask)
        self.t += 1

        if self.joint:
            return self.state
        return tf.Variable(self.state)


    def step_single(self, action):
        segment, b = self.state_ptr[self.t]
        residual = self.ds[segment](self.state) if b==0 else self.state

        if action.data[0,0]==1:
            self.state = tf.nn.relu(residual + self.blocks[segment][b](self.state))
        else:
            self.state = residual

        self.t += 1

        if self.joint:
            return self.state
        return tf.Variable(self.state)

    def predict(self):
        x = self.avgpool(self.state)
        x = tf.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        return x


class StepPolicy32(keras.models.Model):

    def __init__(self, layer_config):
        super(StepPolicy32, self).__init__()
        in_dim = [16] + [16]*layer_config[0] + [32]*layer_config[1] + [64]*(layer_config[2]-1)
        self.pnet = [keras.layers.Dense(2) for dim in in_dim]
        self.vnet = [keras.layers.Dense(1) for dim in in_dim]

    def call(self, state):
        x, t = state
        #x = F.avg_pool2d(x, x.size(2)).view(x.size(0), -1) # pool + flatten --> (B, 16/32/64)
        x = tf.reshape(tf.nn.avg_pool2d(input=x, ksize=x.shape[2], strides=x.shape[2], padding='SAME'),
            [x.shape[0], -1])
        logit = tf.nn.softmax(self.pnet[t](x))
        value = self.vnet[t](x)
        return logit, value