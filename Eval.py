#!/home/local/users/jw/anaconda2/bin/python
# Author: jywang explorerwjy@gmail.com

#=========================================================================
# Evaluation the Results.
#=========================================================================


from datetime import datetime
import math
import time
import sys
import os
import numpy as np
import tensorflow as tf
import Window2Tensor
from Input import *
import Models

BATCH_SIZE = FLAGS.batch_size


def GetCheckPoint():
    ckptfile = FLAGS.checkpoint_dir + '/log/checkpoint'
    if not os.path.isfile(ckptfile):
        print "Model checkpoint not exists."
        exit()
    f = open(ckptfile, 'rb')
    ckpt = f.readline().split(':')[1].strip().strip('"')
    f.close()
    prefix = os.path.abspath(FLAGS.checkpoint_dir + '/log/')
    ckpt = prefix + '/' + ckpt
    return ckpt

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))

def do_eval(sess, eval_correct, data_batch, label_batch,):
    true_count = 0
    steps_per_epoch = Total // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    for step in xrange(steps_per_epoch):
        true_count += sess.run(eval_correct)
    precision = float(true_count) / num_examples
    print '\tNum examples: %d\tNum correct: %d\tPrecision @ 1: %.04f' % (num_examples, true_count, precision)

def runTesting(Data, ModelCKPT):
    DataHand = gzip.open(Data, 'rb')
    DataReader = RecordReader(DataHand)
    # with tf.Graph().as_default() as g:
    with tf.device('/gpu:7'):
        queue_input_data = tf.placeholder(
            dtype, shape=[DEPTH * (HEIGHT + 1) * WIDTH])
        queue_input_label = tf.placeholder(tf.int32, shape=[])
        queue = tf.FIFOQueue(capacity=FLAGS.batch_size * 10,
                                      dtypes=[dtype,
                                              tf.int32],
                                      shapes=[[DEPTH * (HEIGHT + 1) * WIDTH],
                                              []],
                                      name='fifo_queue')
        enqueue_op = queue.enqueue([queue_input_data, queue_input_label])
        dequeue_op = queue.dequeue()
        # Get Tensors and labels for Training data.
        data_batch, label_batch = tf.train.batch(
            dequeue_op, batch_size=FLAGS.batch_size, capacity=FLAGS.batch_size * 4)
        #data_batch_reshape = tf.transpose(data_batch, [0,2,3,1])

        convnets = Models.ConvNets()
        logits = convnets.Inference(data_batch)
        correct = evaluation(logits, label_batch)

        saver = tf.train.Saver()

        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            saver.restore(sess, ModelCKPT)

            # print TrainingLabel
            # print sess.run(logits,feed_dict = {TensorPL:TrainingTensor})

            print "Evaluating On {}}".format(Data)
            stime = time.time()
            do_eval(
                sess,
                correct,
                data_batch,
                label_batch)
            print "Finish Evaluating Testing Dataset. %.3f" % (time.time() - stime)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    # evaluate()
    TrainingData = FLAGS.TrainingData
    ValidationData = FLAGS.ValidationData
    TestingData = FLAGS.TestingData
    #ModelCKPT = FLAGS.checkpoint_dir+'/model.ckpt-4599.meta'
    ModelCKPT = GetCheckPoint()
    runTesting(TrainingData, ValidationData, TestingData, ModelCKPT)


if __name__ == '__main__':
    tf.app.run()
