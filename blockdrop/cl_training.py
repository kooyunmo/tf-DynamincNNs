import os
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import numpy as np
import tqdm
import imagenet_preprocessing
from models.resnet import FlatResNet224, Policy224
from models.base import Bottleneck

# DEBUG OPTION
tf.get_logger().setLevel('ERROR')

import argparse
parser = argparse.ArgumentParser(description='BlockDrop Training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--beta', type=float, default=1e-1, help='entropy multiplier')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--model', default='R110_C10', help='R<depth>_<dataset> see utils.py for a list of configurations')
parser.add_argument('--data_dir', default='data/', help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load agent from')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--lr_decay_ratio', type=float, default=0.1, help='lr *= lr_decay_ratio after epoch_steps')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
parser.add_argument('--cl_step', type=int, default=1, help='steps for curriculum training')
# parser.add_argument('--joint', action ='store_true', default=True, help='train both the policy network and the resnet')
parser.add_argument('--penalty', type=float, default=-1, help='gamma: reward for incorrect predictions')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
args = parser.parse_args()

###############################################################################
# Data processing
###############################################################################

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

def performance_stats(policies, rewards, matches):

    policies = tf.concat(policies, 0) # torch.cat(policies, 0)
    rewards = tf.concat(rewards, 0)
    accuracy = tf.math.reduce_mean(tf.concat(matches, 0)) * 100

    reward = tf.math.reduce_mean(rewards)

    policy_set = [p.numpy().astype(np.int).astype(np.str) for p in policies]
    policy_set = set([''.join(p) for p in policy_set])

    return accuracy, reward, policy_set

def get_reward(preds, targets, policy):

    #block_use = policy.sum(1).float()/policy.size(1)
    block_use = tf.cast(tf.math.reduce_sum(policy, axis=1), tf.float32) / policy.shape[1]
    sparse_reward = 1.0-block_use**2

    #_, pred_idx = preds.max(1)
    pred_idx = tf.math.argmax(preds, axis=1, output_type=tf.dtypes.int32)
    match = (pred_idx==targets)

    reward = sparse_reward
    #reward[tf.cast(tf.logical_not(match), tf.int32)] = args.penalty
    reward = tf.expand_dims(reward, axis=1)

    return reward, tf.cast(match, tf.float32)

#@tf.function
def _one_step(inputs, targets, batch_idx):
    probs, value = agent(inputs)

    policy_map = tf.identity(probs)
    #policy_map[policy_map<0.5] = 0.0
    #policy_map[policy_map>=0.5] = 1.0
    policy_map = tf.cast(policy_map >= 0.5, tf.float32)
    policy_map = tf.Variable(policy_map)

    probs = probs*args.alpha + (1-probs)*(1-args.alpha)
    distr = tfp.distributions.Bernoulli(probs=probs)  #Bernoulli(probs)
    policy = distr.sample()

    if args.cl_step < num_blocks:
        '''
        policy[:, :-args.cl_step] = 1
        policy_map[:, :-args.cl_step] = 1
        '''
        policy_ = tf.transpose(policy)
        p_idx = tf.constant([[i] for i in range(policy_.shape[0]-args.cl_step)])
        p_update = tf.ones_like(policy_[:-args.cl_step,:])
        policy = tf.transpose(tf.tensor_scatter_nd_update(policy_, p_idx, p_update))
        policy_map_ = tf.transpose(policy_map)
        pm_idx = tf.constant([[i] for i in range(policy_map_.shape[0]-args.cl_step)])
        pm_update = tf.ones_like(policy_map_[:-args.cl_step,:])
        policy_map = tf.transpose(tf.tensor_scatter_nd_update(policy_map_, pm_idx, pm_update))

        #policy_mask = Variable(torch.ones(inputs.size(0), policy.size(1))).cuda()
        policy_mask = tf.Variable(tf.ones([inputs.shape[0], policy.shape[1]]))
        policy_mask_ = tf.transpose(policy_mask)
        pmask_idx = tf.constant([[i] for i in range(policy_mask_.shape[0]-args.cl_step)])
        pmask_update = tf.ones_like(policy_mask_[:-args.cl_step,:])
        policy_mask = tf.transpose(tf.tensor_scatter_nd_update(policy_mask_, pmask_idx, pmask_update))
    else:
        policy_mask = None

    v_inputs = tf.Variable(inputs)
    preds_map = rnet.forward(v_inputs, policy_map)
    preds_sample = rnet.forward(v_inputs, policy)

    reward_map, _ = get_reward(preds_map, targets, policy_map)
    reward_sample, match = get_reward(preds_sample, targets, policy)

    advantage = reward_sample - reward_map

    loss = -distr.log_prob(policy)
    #loss = loss * tf.Variable(advantage).expand_as(policy)
    loss = loss * tf.broadcast_to(tf.Variable(advantage), policy.shape)

    if policy_mask is not None:
        loss = policy_mask * loss # mask for curriculum learning

    loss = tf.reduce_sum(loss)

    probs = tf.clip_by_value(probs, clip_value_min=1e-15, clip_value_max=1-1e-15) # probs.clamp(1e-15, 1-1e-15)
    entropy_loss = tf.math.negative(probs) * tf.math.log(probs)   # -probs*torch.log(probs)
    entropy_loss = args.beta * tf.math.reduce_sum(entropy_loss)

    loss = (loss - entropy_loss) / inputs.shape[0]
    
    return loss, match, reward_sample, policy


def train(epoch):

    matches, rewards, policies = [], [], []
    for batch_idx, (inputs, targets) in enumerate(parsed_dataset_train):

        with tf.GradientTape() as tape:
            loss, match, reward_sample, policy = _one_step(inputs, targets, batch_idx)
            
        grads = tape.gradient(loss, agent.trainable_variables)
        optimizer.apply_gradients(zip(grads, agent.trainable_variables))

        matches.append(tf.stop_gradient(match))
        rewards.append(tf.stop_gradient(reward_sample))
        policies.append(tf.stop_gradient(policy))

        compute_loss(loss)

        if batch_idx % 20 == 19:
            accuracy, reward, policy_set = performance_stats(policies, rewards, matches)
            log_str = '[BATCH: %d] | Acc: %.3f | Loss: %.3f | Reward: %.2E | #: %d'%(batch_idx+1, accuracy, compute_loss.result(), reward, len(policy_set))
            print(log_str) 



'''
def test(epoch):

    agent.eval()

    matches, rewards, policies = [], [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        inputs, targets = Variable(inputs, volatile=True), Variable(targets).cuda(async=True)
        if not args.parallel:
            inputs = inputs.cuda()

        probs, _ = agent(inputs)

        policy = probs.data.clone()
        policy[policy<0.5] = 0.0
        policy[policy>=0.5] = 1.0
        policy = Variable(policy)

        if args.cl_step < num_blocks:
            policy[:, :-args.cl_step] = 1

        preds = rnet.forward(inputs, policy)
        reward, match = get_reward(preds, targets, policy.data)

        matches.append(match)
        rewards.append(reward)
        policies.append(policy.data)

    accuracy, reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards, matches)

    log_str = 'TS - A: %.3f | R: %.2E | S: %.3f | V: %.3f | #: %d'%(accuracy, reward, sparsity, variance, len(policy_set))
    print log_str

    log_value('test_accuracy', accuracy, epoch)
    log_value('test_reward', reward, epoch)
    log_value('test_sparsity', sparsity, epoch)
    log_value('test_variance', variance, epoch)
    log_value('test_unique_policies', len(policy_set), epoch)

    # save the model
    agent_state_dict = agent.module.state_dict() if args.parallel else agent.state_dict()

    state = {
      'agent': agent_state_dict,
      'epoch': epoch,
      'reward': reward,
      'acc': accuracy
    }
    torch.save(state, args.cv_dir+'/ckpt_E_%d_A_%.3f_R_%.2E_S_%.2f_#_%d.t7'%(epoch, accuracy, reward, sparsity, len(policy_set)))
'''

#--------------------------------------------------------------------------------------------------------#
#trainset, testset = utils.get_dataset(args.model, args.data_dir)
#trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
#testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)


# constant for ImageNet
batch_size = args.batch_size

# get dataset
data_dir = "/cmsdata/ssd0/cmslab/imagenet-data/"

# ResNet preprocessing
train_filenames = get_filenames(is_training=True, data_dir=data_dir)
raw_dataset_train = tf.data.TFRecordDataset(train_filenames)

test_filenames = get_filenames(is_training=False, data_dir=data_dir)
raw_dataset_test = tf.data.TFRecordDataset(test_filenames)

parsed_dataset_train = raw_dataset_train.map(parse_record).shuffle(1024).batch(batch_size)
parsed_dataset_tests = raw_dataset_test.map(parse_record).shuffle(1024).batch(batch_size)

rnet = FlatResNet224(Bottleneck, [3,4,23,3], 1001, False)
agent = Policy224(layer_config=[1,1,1,1], num_blocks=33, trainable=True)

num_blocks = sum(rnet.layer_config)

start_epoch = 0

compute_loss = tf.keras.metrics.Mean()
compute_acc = tf.keras.metrics.SparseCategoricalAccuracy()

#optimizer = optim.Adam(agent.parameters(), lr=args.lr, weight_decay=args.wd)
optimizer = keras.optimizers.Adam(
    learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=args.lr,
                                                                 decay_steps=args.epoch_step,
                                                                 decay_rate=args.lr_decay_ratio)) 

for epoch in range(start_epoch, start_epoch+args.max_epochs+1):

    if args.cl_step < num_blocks:
        args.cl_step = 1 + 1 * (epoch // 1)
    else:
        args.cl_step = num_blocks

    train(epoch)
    