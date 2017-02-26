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
import numpy as np
import tensorflow as tf


# Basic model parameters.
WIDTH = Region.WIDTH
HEIGHT = Region.HEIGHT
Window_Size = (WIDTH * (HEIGHT+1) * 3)

NUM_CLASSES = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 6400
LEARNING_RATE_DECAY_STEP = 1000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.9  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# Global constants describing the data set & Model.
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', './tmp/TensorCaller_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('train_dir', './tmp/TensorCaller_train',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('log_dir', './tmp/TensorCaller_train/log',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of WindowTensor to process in a batch.""")
tf.app.flags.DEFINE_string('TrainingData', './windows_training.txt.gz',
                           """Path to the Training Data.""")
tf.app.flags.DEFINE_string('ValidationData', './windows_validation.txt.gz',
                           """Path to the Validation Data.""")
tf.app.flags.DEFINE_string('TestingData', './windows_testing.txt.gz',
                           """Path to the Testing Data.""")
tf.app.flags.DEFINE_boolean('use_fl16', True,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# ==========================================================================
def base2code(base):
    try:
        #return tf.cast(BASE[base],tf.float32)
        #return float(BASE[base])
        return float(base)
    except KeyError:
        print "KeyError of base code. Unexpected base appear. |%s|" % base
        exit()
def qual2code(ch):
    phred = (float(ord(ch) - 33) / 60) - 0.5
    #return tf.cast((math.pow(10, -(phred/10))),tf.float32)
    #return float(math.pow(10, -(phred/10)))
    return phred 
def strand2code(ch):
    return float(ch)
# ==========================================================================


class window_tensor():
    def __init__(self,line):
        self.chrom, self.start, self.end, self.label, self.window = line.strip().split('\t')
        self.Alignment = self.window[ 0 : WIDTH * (HEIGHT+1) ]
        self.Qual = self.window[ WIDTH * (HEIGHT+1) : WIDTH * (HEIGHT+1)*2]
        self.Strand = self.window[ WIDTH * (HEIGHT+1)*2 : WIDTH * (HEIGHT+1)*3]

    def encode(self):
        # This func encode,norm elements and form into tensor 
        res = [ (float(base)/6 - 0.5) for base in list(self.Alignment)] + 
              [ qual2code(x) for x in list(self.Qual)] + 
              [ float(x)/2-0.5 for x in list(self.Strand)]
    if FLAGS.use_fl16: 
      RawTensor = tf.convert_to_tensor(res, dtype=tf.float16)
    else:
          RawTensor = tf.convert_to_tensor(res, dtype=tf.float32)
        InputTensor = tf.reshape(RawTensor, [WIDTH, HEIGHT+1, 3]) 
        return InputTensor

class RecordReader():
  def __init__(self, hand):
    self.hand = hand
  def read(self):
    record = window_tensor(self.hand.readline())
    tensor = record.encode()
    label = tf.one_hot(indices=tf.cast(record.label, tf.int16), depth=3)
    return tensor, label

def main():
    return

if __name__=='__main__':
    main()
