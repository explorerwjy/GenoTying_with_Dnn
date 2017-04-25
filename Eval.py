#!/home/yufengshen/anaconda2/bin/python
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
from threading import Thread
from Training import DataReaderThread
import Models

BATCH_SIZE = FLAGS.batch_size
dtype = tf.float16 if FLAGS.use_fl16 else tf.float32

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

def enqueueInputData(
        sess,
        coord,
        Reader,
        enqueue_op,
        queue_input_data,
        queue_input_target):
    try:
        while True:
            curr_data, curr_label = Reader.OnceRead()
            if curr_data == None:
                raise Exception('Finish Reading the file')
            sess.run(
                enqueue_op,
                feed_dict={
                    queue_input_data: curr_data,
                    queue_input_target: curr_label})
    except Exception as e:
        print e
        print("finished enqueueing")
        coord.request_stop(e)

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))

def do_eval(sess, eval_correct, data_batch, label_batch):
    true_count = 0
    num_examples = 0
    while 1:
        true_count += sess.run(eval_correct)
        num_examples += BATCH_SIZE
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
            coord = tf.train.Coordinator()

            enqueue_thread = Thread(
                target=enqueueInputData,
                args=[
                    sess,
                    coord,
                    DataReader,
                    enqueue_op,
                    queue_input_data,
                    queue_input_label])
            enqueue_thread.isDaemon()
            enqueue_thread.start()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            try:
                print "Evaluating On {}".format(Data)
                stime = time.time()
                true_count = 0
                num_examples = 0
                steps = 0
                while 1:
                    true_count += sess.run(correct)
                    num_examples += BATCH_SIZE
                    steps += 1
                    if steps % 10 == 0:
                        print "{} steps run. {} examples read. {} Predicted Correctly. Current Accuracy: {}".format(steps, num_examples, true_count, float(true_count)/num_examples)
            except Exception as e:
                coord.request_stop(e)
                precision = float(true_count) / num_examples
                print '\tNum examples: %d\tNum correct: %d\tPrecision @ 1: %.04f' % (num_examples, true_count, precision)
                print "Finish Evaluating Testing Dataset. %.3f" % (time.time() - stime)
            finally:
                sess.run(queue.close(cancel_pending_enqueues=True))
                coord.request_stop()
                coord.join(threads)

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
    runTesting(TestingData, ModelCKPT)


if __name__ == '__main__':
    tf.app.run()
