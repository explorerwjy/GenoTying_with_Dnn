#!/home/local/users/jw/anaconda2/bin/python
#Author: jywang explorerwjy@gmail.com

#========================================================================================================
# Prepare Input Data For Training
#========================================================================================================

from optparse import OptionParser
import os
import Region
import time
import gzip
import tensorflow as tf

# Basic model parameters.
WIDTH = Region.WIDTH
HEIGHT = Region.HEIGHT
Window_Size = (WIDTH * (HEIGHT+1) * 3)

NUM_CLASSES = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 300000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 30000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


# Global constants describing the data set & Model.
FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('eval_dir', './tmp/TensorCaller_eval',
                           """Directory where to write event logs.""")

tf.app.flags.DEFINE_string('log_dir', './tmp/TensorCaller_train/log',
                           """Directory where to write event logs.""")

tf.app.flags.DEFINE_string('checkpoint_dir', './tmp/TensorCaller_train',
                           """Directory where to read model checkpoints.""")

tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")

tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")

tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of WindowTensor to process in a batch.""")

tf.app.flags.DEFINE_string('TrainingData', './windows_training.txt.gz',
                           """Path to the Training Data.""")

tf.app.flags.DEFINE_string('ValidationData', './windows_validation.txt.gz',
                           """Path to the Validation Data.""")

tf.app.flags.DEFINE_string('TestingData', './windows_testing.txt.gz',
                           """Path to the Testing Data.""")

tf.app.flags.DEFINE_boolean('use_fl16', False,
                            """Train the model using fp16.""")

tf.app.flags.DEFINE_string('train_dir', './tmp/TensorCaller_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('max_steps', 200000,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# ARG: batch_size: The batch size will be baked into both placeholders.
# Return: Tensors placeholder, Labels placeholder.
def placeholder_inputs(batch_size):
  #tensor_placeholder = tf.placeholder(tf.float32, shape=(batch_size,WIDTH,HEIGHT+1,3))
    tensor_placeholder = tf.placeholder(tf.float32, shape=(batch_size,WIDTH*(HEIGHT+1)*3))
    labels_placeholder = tf.placeholder(tf.int32, shape = batch_size)
    return tensor_placeholder, labels_placeholder

def fill_feed_dict(data_set, tensor_pl, labels_pl):
    tensor_feed, labels_feed = data_set.read_batch()
    feed_dict = {
      tensor_pl: tensor_feed,
      labels_pl: labels_feed
      }
    return feed_dict

def main():

    return

if __name__=='__main__':
    main()
