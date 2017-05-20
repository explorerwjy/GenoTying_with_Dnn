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
import Models

GPUs = [7]
available_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([ available_devices[x] for x in GPUs])
print "Using GPU ",os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', './test',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './train_6',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 64000,
                            """Number of examples to run.""")
dtype = tf.float16 if FLAGS.use_fl16 else tf.float32

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

class Evaluate():
    def __init__(self, batch_size, model, DataFile):
        self.batch_size = batch_size
        print "=" * 50
        print "InputData is:", DataFile
        print "=" * 50
        self.DataFile = DataFile
        self.model = model

    def run(self):
        dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
        Hand = gzip.open(self.DataFile, 'rb')
        Reader = RecordReader(Hand)
        with tf.Graph().as_default():
            global_step = tf.Variable(0, trainable=False, name='global_step')
            queue_input_data = tf.placeholder(dtype, shape=[DEPTH * (HEIGHT + 1) * WIDTH])
            queue_input_label = tf.placeholder(tf.int32, shape=[])
            queue = tf.FIFOQueue(capacity=FLAGS.batch_size * 10,
                                      dtypes=[dtype, tf.int32],
                                      shapes=[[DEPTH * (HEIGHT + 1) * WIDTH], []],
                                      name='FIFOQueue')
            enqueue_op = queue.enqueue([queue_input_data, queue_input_label])
            dequeue_op = queue.dequeue()
            # Get Tensors and labels for Training data.
            data_batch, label_batch = tf.train.batch(dequeue_op, batch_size=FLAGS.batch_size, capacity=FLAGS.batch_size * 8)
            logits = self.model.Inference(data_batch)
            loss = self.model.loss(logits, label_batch)
            #accuracy = self.model.Accuracy(logits, label_batch)
            #train_op = self.model.Train(loss, global_step)
            top_k_op = tf.nn.in_top_k(logits, label_batch, 1)

            summary_op = tf.summary.merge_all()
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            sess = tf.Session()
            summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, sess.graph)
            sess.run(init)
            coord = tf.train.Coordinator()
            enqueue_thread = Thread(
                target=enqueueInputData,
                args=[
                    sess,
                    coord,
                    Reader,
                    enqueue_op,
                    queue_input_data,
                    queue_input_label])
            enqueue_thread.isDaemon()
            enqueue_thread.start()
            print "Before Threads"
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            try:    
                num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
                true_count = 0  # Counts the number of correct predictions.
                total_sample_count = num_iter * FLAGS.batch_size
                step = 0
                print self.getCheckPoint()
                saver.restore(sess, self.getCheckPoint())
                print "CKPT starts with step",(sess.run(global_step))
                #for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                #    print(v)
                #    print sess.run("conv1/weights/read:0")
                #exit()

                while step < num_iter and not coord.should_stop():
                    _labels, _logits, _loss, predictions = sess.run([label_batch, logits, loss, top_k_op])
                    #print "labels:",_labels
                    #print "logits:",_logits
                    #print "predict", predictions
                    print "loss:",_loss
                    true_count += np.sum(predictions)
                    step += 1
                
                # Compute precision @ 1.
                print "Predicted Right:{}\t\tTotal:{}".format(true_count, total_sample_count)
                precision = float(true_count) / total_sample_count
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
                #summary = tf.Summary()
                #summary.ParseFromString(sess.run(summary_op))
                #summary.value.add(tag='Precision @ 1', simple_value=precision)
                #summary_writer.add_summary(summary, step)
            except Exception, e:
                coord.request_stop(e)
            finally:
                print "Predicted Right:{}\t\tTotal:{}".format(true_count, total_sample_count)
                precision = float(true_count) / total_sample_count
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
                sess.run(queue.close(cancel_pending_enqueues=True))
                coord.request_stop()
                coord.join(threads)

    def getCheckPoint(self):
        ckptfile = FLAGS.checkpoint_dir + '/checkpoint'
        f = open(ckptfile, 'rb')
        ckpt = f.readline().split(':')[1].strip().strip('"')
        f.close()
        prefix = os.path.abspath(FLAGS.checkpoint_dir)
        ckpt = prefix + '/' + ckpt
        return ckpt

def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    #DataFile = FLAGS.TrainingData
    DataFile = FLAGS.TestingData
    model = Models.ConvNets()
    evaluate = Evaluate(FLAGS.batch_size, model, DataFile)
    evaluate.run()


if __name__ == '__main__':
    tf.app.run()
