import numpy as np
import tensorflow as tf
import models
import time
import os

import inception_preprocessing
import imagenet_preprocessing

import logging
import datetime
suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
logging.basicConfig(filename='eager_'+suffix+".log", level=logging.INFO)

tf.random.set_seed(0)
np.random.seed(0)

#tf.keras.backend.set_image_data_format('channels_first')

DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_CLASSES = 1001

NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 10000

DATASET_NAME = 'ImageNet'

###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
    """Return filenames for dataset."""
    if is_training:
        return [
            os.path.join(data_dir, 'train-%05d-of-01024' % i)
            for i in range(_NUM_TRAIN_FILES)]
    else:
        return [
            os.path.join(data_dir, 'validation-%05d-of-00128' % i)
            for i in range(128)]


def _parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.
    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields (values are included as examples):
        image/height: 462
        image/width: 581
        image/colorspace: 'RGB'
        image/channels: 3
        image/class/label: 615
        image/class/synset: 'n03623198'
        image/class/text: 'knee pad'
        image/object/bbox/xmin: 0.1
        image/object/bbox/xmax: 0.9
        image/object/bbox/ymin: 0.2
        image/object/bbox/ymax: 0.6
        image/object/bbox/label: 615
        image/format: 'JPEG'
        image/filename: 'ILSVRC2012_val_00041207.JPEG'
        image/encoded: <JPEG encoded string>
    Args:
        example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.
    Returns:
        image_buffer: Tensor tf.string containing the contents of a JPEG file.
        label: Tensor tf.int32 containing the label.
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
    """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                                default_value=''),
        'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64,
                                                    default_value=-1),
        'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                                    default_value=''),
    }
    
    sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update(
        {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                     'image/object/bbox/ymin',
                                     'image/object/bbox/xmax',
                                     'image/object/bbox/ymax']})

    features = tf.io.parse_single_example(serialized=example_serialized,
                                            features=feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(a=bbox, perm=[0, 2, 1])

    return features['image/encoded'], label, bbox


def parse_record(raw_record, is_training=True, dtype=tf.float32):
    """Parses a record containing a training example of an image.
    The input record is parsed into a label and image, and the image is passed
    through preprocessing steps (cropping, flipping, and so on).
    Args:
        raw_record: scalar Tensor tf.string containing a serialized
        Example protocol buffer.
        is_training: A boolean denoting whether the input is for training.
        dtype: data type to use for images/features.
    Returns:
        Tuple with processed image tensor and one-hot-encoded label tensor.
    """
    image_buffer, label, bbox = _parse_example_proto(raw_record)

    image = imagenet_preprocessing.preprocess_image(
        image_buffer=image_buffer,
        bbox=bbox,
        output_height=DEFAULT_IMAGE_SIZE,
        output_width=DEFAULT_IMAGE_SIZE,
        num_channels=NUM_CHANNELS,
        is_training=is_training
    )
    image = tf.cast(image, dtype)

    return image, label

################################################################

class ListAverageMeter(object):
    """Computes and stores the average and current values of a list"""

    def __init__(self):
        self.len = 10000  # set up the maximum length
        self.reset()

    def reset(self):
        self.val = [0] * self.len
        self.avg = [0] * self.len
        self.sum = [0] * self.len
        self.count = 0

    def set_len(self, n):
        self.len = n
        self.reset()

    def update(self, vals, n=1):
        assert len(vals) == self.len, 'length of vals not equal to self.len'
        self.val = vals
        for i in range(self.len):
            self.sum[i] += self.val[i] * n
        self.count += n
        for i in range(self.len):
            self.avg[i] = self.sum[i] / self.count

#@tf.function
def _one_step(model, x, y, skip_ratios):
    '''
    ouput: tf.Tensor: shape=(BATCH_SIZE,1000)
    masks: array of tf.Tensors: shape=(BATCH_SIZE,)
    probs: array of tf.Tensors: shape=(BATCH_SIZE,)
    hidden: tuple(tf.Tensor: shape=(1,BATCH_SIZE,10), tf.Tensor: shape=(1,BATCH_SIZE,10))
    '''
    output, masks, probs, hidden = model(x, training=True)
    #output = model(x, training=True)

    #skips = [mask.data.le(0.5).float().mean() for mask in masks]
    '''
    skips = [tf.reduce_mean(tf.cast((mask <= 0.5), tf.float32)) for mask in masks]
    if skip_ratios.len != len(skips):
        skip_ratios.set_len(len(skips))
    '''
    
    #loss = criterion(output, target_var)
    loss = tf.nn.softmax_cross_entropy_with_logits(y, output)

    '''
    #skip_ratios.update(skips, input.size(0))
    skip_ratios.update(skips, x.shape[0])
    '''

    return output, loss

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.Accuracy(name='train_acc')

# Trains the model for certains epochs on a dataset
def train(dset_train, dset_test, model, epochs=5, show_loss=False):
    
    for epoch in range(epochs):
        skip_ratios = ListAverageMeter()
        time_sum = 0
        cnt = 0
        for x, y in dset_train: # for every batch
            y = tf.one_hot(y, 1001)
            start = time.time()
            with tf.GradientTape() as g:
                preds, loss = _one_step(model, x, y, skip_ratios)
            
            grads = g.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            time_sum += time.time() - start
            cnt += 1

            train_loss(loss)
            train_acc(tf.argmax(y, 1), tf.argmax(preds, 1))

            if cnt % 50 == 49:
        	    print("[BATCH {}]avg elapsed time: {}sec/step || loss: {} || acc: {}".format(cnt, time_sum / cnt,
                                                                                             train_loss.result(),
                                                                                             train_acc.result() * 100))

        print("[EPOCH {}/{}] avg elapsed time: {}sec/step || loss: {} || acc: {}".format(epoch+1, epochs,
                                                                                         time_sum / cnt,
                                                                                         train_loss.result(),
                                                                                         train_acc.result() * 100))

		# Get accuracies
		#train_acc = get_accuracy(dset_train, model, training=True)
		#test_acc = get_accuracy(dset_test, model, writer=writer_test)
		# write summaries and print
		#write_summary(train_acc, writer_train, 'accuracy')
		#write_summary(test_acc, writer_test, 'accuracy')
		#print('Train accuracy: ' + str(train_acc.numpy()))
		#print('Test accuracy: ' + str(test_acc.numpy()))


# Tests the model on a dataset
'''
def get_accuracy(dset_test, model, training=False,  writer=None):
	accuracy = tfe.metrics.Accuracy()
	if writer: loss = [0, 0]

	for x, y in dset_test: # for every batch
		y_ = model(x, training=training)
		accuracy(tf.argmax(y, 1), tf.argmax(y_, 1))

		if writer:
			loss[0] += tf.losses.softmax_cross_entropy(y, y_)
			loss[1] += 1.

	if writer:
		write_summary(tf.convert_to_tensor(loss[0]/loss[1]), writer, 'loss')

	return accuracy.result()
'''

def restore_state(saver, checkpoint):
	try:
		saver.restore(checkpoint)
		print('Model loaded')
	except Exception:
		print('Model not loaded')


def init_model(model, input_shape):
	model._set_inputs(np.zeros(input_shape))

if __name__ == "__main__":

    # constants for imagenet
    batch_size = 32     # TITAN XP: OOM with batch size 64
    epochs = 1
    image_size = 224
    channels = 3
    classes = 1001

    # Get dataset
    data_dir = "/cmsdata/ssd0/cmslab/imagenet-data/"

    # ResNet preprocessing
    train_filenames = get_filenames(is_training=True, data_dir=data_dir)
    raw_dataset_train = tf.data.TFRecordDataset(train_filenames)

    test_filenames = get_filenames(is_training=False, data_dir=data_dir)
    raw_dataset_test = tf.data.TFRecordDataset(test_filenames)

    # HY: This dataset is for resnet preprocessing
    parsed_dataset_train = raw_dataset_train.map(parse_record).shuffle(1024).batch(batch_size)
    parsed_dataset_tests = raw_dataset_test.map(parse_record).shuffle(1024).batch(batch_size)

    # dataset_train = imagenet.get_split('train', data_dir)
    # dataset_tests = imagenet.get_split('test', data_dir)

    model = models.imagenet_rnn_gate_50()
    #model = models.resnet50()

    # optimizer
    #optimizer = tf.train.AdamOptimizer(0.001)
    optimizer = tf.keras.optimizers.Adam()

    print("############ Start Training ##############")
    logging.info("############ Start Training ##############")
    train(dset_train=parsed_dataset_train, dset_test=parsed_dataset_tests, model=model, epochs=epochs)
