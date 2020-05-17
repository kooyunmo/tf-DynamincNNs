# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app as absl_app
from absl import flags
from absl import logging
from six.moves import range
import tensorflow as tf

from official.r1.resnet import imagenet_preprocessing
from official.r1.resnet import resnet_model
from official.r1.resnet import resnet_run_loop
from official.r1.utils.logs import logger
from official.utils.flags import core as flags_core

# added for inception preprocessing
# from research.slim.preprocessing import inception_preprocessing
# modified inception_preprocessing for tfv2
import inception_preprocessing

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
    # This is from inception_preprocessing
    # feature_map = {
    #     'image/encoded': tf.io.FixedLenFeature(
    #         (), tf.string, default_value=''),
    #     'image/format': tf.io.FixedLenFeature(
    #         (), tf.string, default_value='jpeg'),
    #     'image/class/label': tf.io.FixedLenFeature(
    #         [], dtype=tf.int64, default_value=-1),
    #     'image/class/text': tf.io.FixedLenFeature(
    #         [], dtype=tf.string, default_value=''),
    #     'image/object/bbox/xmin': tf.io.VarLenFeature(
    #         dtype=tf.float32),
    #     'image/object/bbox/ymin': tf.io.VarLenFeature(
    #         dtype=tf.float32),
    #     'image/object/bbox/xmax': tf.io.VarLenFeature(
    #         dtype=tf.float32),
    #     'image/object/bbox/ymax': tf.io.VarLenFeature(
    #         dtype=tf.float32),
    #     'image/object/class/label': tf.io.VarLenFeature(
    #         dtype=tf.int64),
    # }
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

    # image = imagenet_preprocessing.preprocess_image(
    #     image_buffer=image_buffer,
    #     bbox=bbox,
    #     output_height=DEFAULT_IMAGE_SIZE,
    #     output_width=DEFAULT_IMAGE_SIZE,
    #     num_channels=NUM_CHANNELS,
    #     is_training=is_training)

    #HY: inception preprocessing rather than resnet preprocessing
    image = inception_preprocessing.preprocess_image(
        image = image_buffer,
        height = DEFAULT_IMAGE_SIZE,
        width = DEFAULT_IMAGE_SIZE,
        is_training = is_training,
        bbox = bbox
    )
    image = tf.cast(image, dtype)

    return image, label


def input_fn(is_training,
             data_dir,
             batch_size,
             num_epochs=1,
             dtype=tf.float32,
             datasets_num_private_threads=None,
             parse_record_fn=parse_record,
             input_context=None,
             drop_remainder=False,
             tf_data_experimental_slack=False):
    """Input function which provides batches for train or eval.
    Args:
        is_training: A boolean denoting whether the input is for training.
        data_dir: The directory containing the input data.
        batch_size: The number of samples per batch.
        num_epochs: The number of epochs to repeat the dataset.
        dtype: Data type to use for images/features
        datasets_num_private_threads: Number of private threads for tf.data.
        parse_record_fn: Function to use for parsing the records.
        input_context: A `tf.distribute.InputContext` object passed in by
        `tf.distribute.Strategy`.
        drop_remainder: A boolean indicates whether to drop the remainder of the
        batches. If True, the batch dimension will be static.
        tf_data_experimental_slack: Whether to enable tf.data's
        `experimental_slack` option.
    Returns:
        A dataset that can be used for iteration.
    """
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if input_context:
        logging.info(
            'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d',
            input_context.input_pipeline_id, input_context.num_input_pipelines)
        dataset = dataset.shard(input_context.num_input_pipelines,
                                input_context.input_pipeline_id)

    if is_training:
        # Shuffle the input files
        dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

    # Convert to individual records.
    # cycle_length = 10 means that up to 10 files will be read and deserialized in
    # parallel. You may want to increase this number if you have a large number of
    # CPU cores.
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=10,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return resnet_run_loop.process_record_dataset(
        dataset=dataset,
        is_training=is_training,
        batch_size=batch_size,
        shuffle_buffer=_SHUFFLE_BUFFER,
        parse_record_fn=parse_record_fn,
        num_epochs=num_epochs,
        dtype=dtype,
        datasets_num_private_threads=datasets_num_private_threads,
        drop_remainder=drop_remainder,
        tf_data_experimental_slack=tf_data_experimental_slack,
    )


def get_synth_input_fn(dtype):
    return resnet_run_loop.get_synth_input_fn(
        DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, NUM_CHANNELS, NUM_CLASSES,
        dtype=dtype)