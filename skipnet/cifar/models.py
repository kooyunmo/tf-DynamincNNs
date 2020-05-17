import math
import tensorflow as tf
import tensorflow.keras as keras
from threading import Lock

import time

global_lock = Lock()

def conv3x3(in_planes, out_planes, stride=1):
    '''3x3 convolution with padding'''
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
        self.conv2 = keras.layers.Conv2D(planes, kernel_size=(3, 3), strides=(stride, stride),
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

class ResNet(keras.models.Model):
    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 16) 
        self.bn1 = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
        
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = keras.layers.AveragePooling2D(pool_size=(8, 8))   # nn.AvgPool2d(7)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = keras.layers.Dense(num_classes)

        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        '''

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = keras.Sequential([
                keras.layers.Conv2D(planes * block.expansion, kernel_size=(1, 1),
                                    strides=(stride, stride), use_bias=False),
                keras.layers.BatchNormalization()
            ])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return keras.Sequential(layers)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = tf.reshape(x, [x.shape[0], -1])   # x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# For CIFAR-10
# ResNet-38
def cifar10_resnet_38(pretrained=False, **kwargs):
    # n = 6
    model = ResNet(BasicBlock, [6, 6, 6], **kwargs)
    return model


# ResNet-74
def cifar10_resnet_74(pretrained=False, **kwargs):
    # n = 12
    model = ResNet(BasicBlock, [12, 12, 12], **kwargs)
    return model


# ResNet-110
def cifar10_resnet_110(pretrained=False, **kwargs):
    # n = 18
    model = ResNet(BasicBlock, [18, 18, 18], **kwargs)
    return model


# ResNet-152
def cifar10_resnet_152(pretrained=False, **kwargs):
    # n = 25
    model = ResNet(BasicBlock, [25, 25, 25], **kwargs)
    return model


######## train test with CIFAR-10 ########
tf.random.set_seed(0)

cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

BATCH_SIZE = 256

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')

model = cifar10_resnet_74()

@tf.function
def _one_step(inputs, labels):
    preds = model(inputs)
    loss = loss_object(labels, preds)
    return preds, loss

def test_step(images, labels):
    preds = model(images)
    t_loss = loss_object(labels, preds)

    test_loss(t_loss)
    test_acc(labels, preds)

EPOCHS = 5

for epoch in range(EPOCHS):
    time_sum = 0
    cnt = 0
    for images, labels in train_dataset:
        start = time.time()

        with tf.GradientTape() as tape:
            preds, loss = _one_step(images, labels)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        time_sum += time.time() - start
        cnt += 1

        train_loss(loss)
        train_acc(labels, preds)
    
        if cnt % 100 == 99:
            print("avg elapsed time per step: ", time_sum / cnt)

    template = "[EPOCH {}/{}] loss: {}\t acc: {}\t avg elapsed time: {} sec/step"
    print(template.format(epoch + 1,
                          EPOCHS,
                          train_loss.result(),
                          train_acc.result() * 100,
                          time_sum / cnt))