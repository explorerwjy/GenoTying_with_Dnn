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
HEIGHT = Region.HEIGHT
DEPTH = 3
Window_Size = (WIDTH * (HEIGHT + 1) * 3)

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


tf.app.flags.DEFINE_integer('batch_size', 64,
    """Number of WindowTensor to process in a batch.""")

tf.app.flags.DEFINE_string('TrainingData', './Training.windows.txt.gz',
                           """Path to the Training Data.""")

tf.app.flags.DEFINE_string('ValidationData', './Validation.windows.txt.gz',
                           """Path to the Validation Data.""")

tf.app.flags.DEFINE_string('TestingData', 'Testing.windows.txt.gz',
                           """Path to the Testing Data.""")

tf.app.flags.DEFINE_boolean('use_fl16', False,
                            """Train the model using fp16.""")

tf.app.flags.DEFINE_boolean('numOfDecodingThreads', 4,
                            """Whether to log device placement.""")

npdtype = np.float16 if FLAGS.use_fl16 else np.float32


class window_tensor():
    def __init__(self, line):
        self.chrom, self.start, self.end, self.ref, self.alt, self.label, self.window = line.strip().split('\t')
        #self.Alignment = self.window[ 0 : WIDTH * (HEIGHT+1) ]
        #self.Qual = self.window[ WIDTH * (HEIGHT+1) : WIDTH * (HEIGHT+1)*2]
        #self.Strand = self.window[ WIDTH * (HEIGHT+1)*2 : WIDTH * (HEIGHT+1)*3]

    def encode(self):
        # This func encode elements in window tensor into tf.float32
        #p1 = np.fromiter(self.Alignment, dtype = npdtype)
        #p2 = np.array(map(lambda x: qual2code(x), self.Qual), dtype = npdtype)
        #p3 = np.fromiter(self.Strand, dtype = npdtype)
        #self.res = np.concatenate([p1, p2, p3])
        self.res = self.chrom, self.start, self.ref, self.alt, decodeline.DecodeLine(
            self.window, WIDTH, HEIGHT)
        # return np.array(res)


class RecordReader():
    def __init__(self, handle):
        self.hand = handle

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
    
    def OnceRead(self):
        line = self.hand.readline()
        if line == '':
            return None, None 
        return decodeline.DecodeRecord(line, WIDTH, HEIGHT)

    def OnceReadWithInfo(self):
        line = self.hand.readline()
        if line == '':
            return None, None, None, None, None, None # one_tensor, chrom, pos, ref, alt, label
        return decodeline.DecodeRecord3(line, WIDTH, HEIGHT)

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

    def read3(self):
        tensor, chroms, starts, refs, alts, labels = [], [], [], [], [], []
        for i in xrange(FLAGS.batch_size):
            line = self.hand.readline()
            if line == '':
                break
            one_tensor, chrom, pos, ref, alt, label = decodeline.DecodeRecord3(line, WIDTH, HEIGHT)
            tensor.append(one_tensor)
            chroms.append(chrom)
            starts.append(pos)
            refs.append(ref)
            alts.append(alt)
            labels.append(label)
        return tensor, chroms, starts, refs, alts, labels

    def read_without_processing(self):
        line = self.hand.readline()
        if line == '':
            self.hand.seek(0)
            line = self.hand.readline()
        record = window_tensor(self.hand.readline())
# ==========================================================================
#@vectorize(["float32(chr)"], target='gpu')


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
