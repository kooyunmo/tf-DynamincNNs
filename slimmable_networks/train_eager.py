import argparse
import os
import shutil
import time
import tensorflow as tf

from models import s_resnet
import imagenet_preprocessing

# DEBUG OPTION
tf.get_logger().setLevel('ERROR')


parser = argparse.ArgumentParser(description='TensorFlow ImageNet Training - Stochastic Downsampling')
parser.add_argument('--epochs', default=115, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-vb', '--val-batch-size', default=1024, type=int,
                    metavar='N', help='validation mini-batch size (default: 1024)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--val-results-path', default='val_results.txt', type=str,
                    help='filename of the file for writing validation results')

best_prec1 = 0

tf.random.set_seed(0)

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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


'''
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after the 40th, 75th, and 105th epochs"""
    lr = args.lr
    for e in [40,75,105]:
        if epoch >= e:
            lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
'''


@tf.function
def _one_step(model, x, y, criterion):
    logits = model(x, training=True)
    loss = criterion(y, logits)

    return logits, loss

def train(dset_train, dset_test, model, criterion, optimizer, epoch, accuracy, compute_loss):
    step_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    for i, (input, target) in enumerate(dset_train):
        # measure data loading time
        data_time.update(time.time() - end)

        with tf.GradientTape() as tape:
            preds, loss = _one_step(model, input, target, criterion)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # measure accuracy and record loss
        compute_loss(loss)
        accuracy(target, preds)

        # measure elapsed time
        step_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('[Epoch: {}][Batch: {}]\t Loss {}\t acc {}'.format(epoch, i,
                                                                     compute_loss.result(),
                                                                     accuracy.result() * 100))


def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    model = s_resnet.Model()

    # define loss function (criterion) and optimizer
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # nn.CrossEntropyLoss().cuda()

    #optimizer = torch.optim.SGD(model.parameters(),
    #							 args.lr, momentum=args.momentum,
    # 							 weight_decay=args.weight_decay)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(args.lr, 10000, 0.7),
        momentum=args.momentum)

    compute_loss = tf.keras.metrics.Mean()
    compute_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    compute_loss.reset_states()
    compute_acc.reset_states()

    # constant for ImageNet
    batch_size = 32

    # get dataset
    data_dir = "/cmsdata/ssd0/cmslab/imagenet-data/"

    # ResNet preprocessing
    train_filenames = get_filenames(is_training=True, data_dir=data_dir)
    raw_dataset_train = tf.data.TFRecordDataset(train_filenames)

    test_filenames = get_filenames(is_training=False, data_dir=data_dir)
    raw_dataset_test = tf.data.TFRecordDataset(test_filenames)

    # HY: This dataset is for resnet preprocessing
    parsed_dataset_train = raw_dataset_train.map(parse_record).shuffle(1024).batch(batch_size)
    parsed_dataset_tests = raw_dataset_test.map(parse_record).shuffle(1024).batch(batch_size)

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        #train(train_loader, model, criterion, optimizer, epoch)
        train(dset_train=parsed_dataset_train, dset_test=parsed_dataset_tests, model=model,
              criterion=criterion, optimizer=optimizer, epoch=epoch,
              accuracy=compute_acc, compute_loss=compute_loss)


if __name__ == '__main__':
    main()
