#!/home/local/users/jw/anaconda2/bin/python
# Author: jywang explorerwjy@gmail.com

#=========================================================================
# Prepare Input Data For Training
#=========================================================================

from optparse import OptionParser
import os
import Region
import time
import gzip
import numpy as np
import decodeline
import tensorflow as tf

# Basic model parameters.
WIDTH = Region.WIDTH
HEIGHT = Region.HEIGHT + 1
DEPTH = 3
#Window_Size = (WIDTH * (HEIGHT) * 3)

NUM_CLASSES = 3
#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 6400
LEARNING_RATE_DECAY_STEP = 50000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
#NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 1e-4       # Initial learning rate.


# Global constants describing the data set & Model.
#FLAGS = tf.app.flags.FLAGS

#npdtype = np.float16 if FLAGS.use_fl16 else np.float32
npdtype = np.float32


class RecordReader():
    def __init__(self, handle):
        self.hand = handle

    # Used for training, keep reading the data.
    def LoopRead(self):
        line = self.hand.readline()
        if line == '':
            self.hand.seek(0)
            line = self.hand.readline()
        #record = window_tensor(line)
        # record.encode()
        # print len(record.res)
        # return record.res, record.label
        return decodeline.DecodeRecord(line, WIDTH, HEIGHT)
    
    #Used When Evaluating
    def OnceRead(self):
        line = self.hand.readline()
        if line == '':
            return None, None 
        return decodeline.DecodeRecord(line, WIDTH, HEIGHT)

    #Used When Calling.
    def OnceReadWithInfo(self):
        line = self.hand.readline()
        if line == '':
            return None, None, None, None, None, None # one_tensor, chrom, pos, ref, alt, label
        return decodeline.DecodeRecord_WithInfo(line, WIDTH, HEIGHT)

    def read2(self):
        tensor, chroms, starts, refs, alts = [], [], [], [], []
        for i in xrange(FLAGS.batch_size):
            line = self.hand.readline()
            if line == '':
                break
            one_tensor, chrom, pos, ref, alt = decodeline.DecodeRecord2(
                line, WIDTH, HEIGHT)
            tensor.append(one_tensor)
            chroms.append(chrom)
            starts.append(pos)
            refs.append(ref)
            alts.append(alt)

        return tensor, chroms, starts, refs, alts

    # Batch Version of OnceReadWithInfo(). Currently Not Working because unsolved Queue issue.
    def read3(self):
        tensor, chroms, starts, refs, alts, labels = [], [], [], [], [], []
        for i in xrange(FLAGS.batch_size):
            line = self.hand.readline()
            if line == '': # Read up All Records. Compensate the rest to make complete record.
                #fake_tensor, fake_chrom, fake_pos, fake_ref, fake_alt, fake_label = tensor[-1], chroms[-1], starts[-1], refs[-1], alts[-1], labels[-1]
                one_tensor, chrom, pos, ref, alt, label = tensor[-1], ".", ".", ".", ".", "0"
            else:
                one_tensor, chrom, pos, ref, alt, label = decodeline.DecodeRecord_WithInfo(line, WIDTH, HEIGHT)
            tensor.append(one_tensor)
            chroms.append(chrom)
            starts.append(pos)
            refs.append(ref)
            alts.append(alt)
            labels.append(label)
        return tensor, chroms, starts, refs, alts, labels

# ==========================================================================

def base2code(base):
    try:
        # return tf.cast(BASE[base],tf.float32)
        # return float(BASE[base])
        return float(base)
    except KeyError:
        print "KeyError of base code. Unexpected base appear. |%s|" % base
        exit()


def qual2code(ch):
    phred = (float(ord(ch) - 33) / 60) - 0.5
    # return tf.cast((math.pow(10, -(phred/10))),tf.float32)
    # return float(math.pow(10, -(phred/10)))
    return phred


def strand2code(ch):
    return float(ch)
# ==========================================================================


def TestReadingTime():
    Hand = gzip.open(FLAGS.TrainingData, 'rb')
    Reader = RecordReader(Hand)
    CompareSteps = 128
    print 'Reading with decoding'
    count = 0
    s_time = time.time()
    while count < CompareSteps:
        Reader.read()
        count += 1
    print "With Decoding: ", time.time() - s_time
    print "\nReading without Decoding..."
    s_time = time.time()
    count = 0
    while count < CompareSteps:
        Reader.read_without_processing()
        count += 1
    print "Without Decoding: ", time.time() - s_time


def main():
    TestReadingTime()
    return


if __name__ == '__main__':
    main()
