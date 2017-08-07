#!/home/yufengshen/anaconda2/bin/python
# Author: jywang explorerwjy@gmail.com

#=========================================================================
# Training The ResNet for Tensor Caller
#=========================================================================

import argparse
from datetime import datetime
import time
import os
from threading import Thread
import numpy as np
import tensorflow as tf
from Input import *
import sys
import pysam
from collections import deque
from ResNet import * 

sys.stdout = sys.stderr
MOMENTUM = 0.9

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 1e-4, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 50000000, "max steps")
tf.app.flags.DEFINE_boolean('resume', False,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('learning_rate_decay', False,
                            'Whether decay learning rate')
tf.app.flags.DEFINE_boolean('learning_rate_decay_step', 200000,
                            'Frequency to decay the learning rate')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'How many GPUs to use.')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')

GPUs = [2]
available_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([ available_devices[x] for x in GPUs])
print "Using GPU ",os.environ['CUDA_VISIBLE_DEVICES']
#init_lr = FLAGS.learning_rate
EVAL_NUM = 1000 # Num of training data to form a accuracy evaluation
#EVAL_NUM = 320 # Num of training data to form a accuracy evaluation
init_lr = 1e-6
#optimizer = 'RMSProp'
optimizer = 'Adam'
#NUM_BLOCKS = [3, 4, 6, 3] # This is the default 50-layer network
NUM_BLOCKS = [3, 4, 6, 3] 
USE_BIAS = True
BOTTLENECK = True
print "Optimizer is {}, init learning rate is {}. ConV weight loss is {}. FC weight loss is {}. DropoutKeepProp is {}.".format(optimizer, init_lr, CONV_WEIGHT_DECAY, FC_WEIGHT_DECAY, Keep_Prop)
print "num_blocks:{}, use_bias:{}, bottleneck:{}".format(', '.join(map(str, NUM_BLOCKS)), str(USE_BIAS), str(BOTTLENECK))
def enqueueInputData(
        sess,
        coord,
        Reader,
        enqueue_op,
        queue_input_data,
        queue_input_target):
    try:
        while True:
            curr_data, curr_label = Reader.LoopRead()
            sess.run(
                enqueue_op,
                feed_dict={
                    queue_input_data: curr_data,
                    queue_input_target: curr_label})
    except Exception as e:
        print e
        print("finished enqueueing")
        coord.request_stop(e)

class LossQueue:
    def __init__(self):
        self.queue = deque([500] * 100, 100)
    def enqueue(self, value):
        self.queue.appendleft(value)
        self.queue.pop()
    def avgloss(self):
        res = list(self.queue)
        return float(sum(res))/len(res)

class AccuracyQueue:
    def __init__(self, batch_size, batch_num):
        self.AccuracyQueue = deque([0] * 100, 100)
        self.batch_num = batch_num
        self.batch_size = batch_size
        self.Correct = 0.0
        self.Total = 0.0
        self.counter = 0
    def update(self, correct, step):
        if self.counter == self.batch_num:
            Accuracy = self.Correct/self.Total
            print '@ Step {}: \t {} in {} Correct, Batch precision @ 1 ={}'.format(step, self.Correct, self.Total, Accuracy)
            tf.summary.scalar('TrainingAccuracy', Accuracy)
            self.AccuracyQueue.appendleft(Accuracy)
            self.AccuracyQueue.pop()
            self.counter = 0
            self.Total = 0
            self.Correct = 0
        self.Total += self.batch_size
        self.Correct += correct
        self.counter += 1
    def checkQueue(self):
        return (self.Accuracy[0] > self.Accuracy[-1])


class Train():
    def __init__(self, batch_size, model, TrainingDataFile, TestingDataFile):
        self.TrainingDataFile = TrainingDataFile
        self.TestingDataFile = TestingDataFile
        self.batch_size = batch_size
        self.model = model
        print "TrainingData:", TrainingDataFile
        print "Batch Size:", self.batch_size
        print "Train/Log Dir", FLAGS.train_dir

    def run(self, continueModel=None):
        dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
        TrainingHand = gzip.open(self.TrainingDataFile, 'rb')
        TrainingReader = RecordReader(TrainingHand)
        with tf.Graph().as_default():
            queue_input_data = tf.placeholder(dtype, shape=[DEPTH * (HEIGHT) * WIDTH])
            queue_input_label = tf.placeholder(tf.int32, shape=[])
            queue = tf.RandomShuffleQueue(capacity=FLAGS.batch_size * 10,
                                      dtypes=[dtype, tf.int32],
                                      shapes=[[DEPTH * (HEIGHT) * WIDTH], []],
                                      min_after_dequeue=FLAGS.batch_size,
                                      name='RandomShuffleQueue')
            enqueue_op = queue.enqueue([queue_input_data, queue_input_label])
            dequeue_op = queue.dequeue()
            # Get Tensors and labels for Training data.
            data_batch, label_batch = tf.train.batch(dequeue_op, batch_size=FLAGS.batch_size, capacity=FLAGS.batch_size * 8)
            
            #data_batch_reshape = tf.transpose(data_batch, [0,2,3,1])

            global_step = tf.Variable(0, trainable=False, name='global_step')
            #logits = self.model.Inference(data_batch)
            logits = self.model.Inference(data_batch, num_blocks=NUM_BLOCKS, use_bias=USE_BIAS, bottleneck=BOTTLENECK)
            loss = self.model.loss(logits, label_batch)
            #accuracy = self.model.Accuracy(logits, label_batch)
            train_op = self.model.Train(loss, global_step, FLAGS.learning_rate, optimizer)
            top_k_op = tf.nn.in_top_k(logits, label_batch, 1)
            summary_op = tf.summary.merge_all()
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            sess = tf.Session()
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
            sess.run(init)
            print "Before Queue"
            # Start the queue runners.
            coord = tf.train.Coordinator()
            enqueue_thread = Thread(
                target=enqueueInputData,
                args=[
                    sess,
                    coord,
                    TrainingReader,
                    enqueue_op,
                    queue_input_data,
                    queue_input_label])
            enqueue_thread.isDaemon()
            enqueue_thread.start()
            print "Before Threads"
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            loss_queue = LossQueue() 
            min_loss = loss_queue.avgloss()
            try:    
                print "Start"
                training_accuracy = AccuracyQueue(self.batch_size, EVAL_NUM)
                if continueModel != None:
                    saver.restore(sess, continueModel)
                    print "Continue Train Mode. Start with step",sess.run(global_step) 
                    
                for step in xrange(FLAGS.max_steps):
                    if coord.should_stop():
                        break
                    start_time = time.time()

                    _, loss_value, v_step = sess.run([train_op, loss, global_step])
                    duration = time.time() - start_time

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    
                    prediction = float((np.sum(sess.run(top_k_op))))
                    training_accuracy.update(prediction, v_step)


                    if v_step % 10 == 0:
                        loss_queue.enqueue(loss_value)
                        #avgloss = loss_queue.avgloss()
                        num_examples_per_step = FLAGS.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration / FLAGS.num_gpus
                        format_str = ('%s: step %d, loss = %.5f (%.1f examples/sec; %.3f '
                          'sec/batch)')
                        print (format_str % (time.time(), v_step, loss_value,
                                 examples_per_sec, sec_per_batch))
                    
                    if v_step % 100 == 0:
                        #prediction = float((np.sum(sess.run(top_k_op))))
                        #print '@ Step {}: \t {} in {} Correct, Batch precision @ 1 ={}'.format(v_step, prediction, self.batch_size, prediction/self.batch_size)
                        #print accuracy
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, v_step)

                    # Save the model checkpoint periodically.
                    if v_step % 1000 == 0 or (v_step + 1) == FLAGS.max_steps:
                        #self.EvalWhileTraining()
                        avgloss = loss_queue.avgloss()
                        if avgloss < min_loss:
                            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                            saver.save(sess, checkpoint_path, global_step=v_step)
                            min_loss = avgloss 
                            print "Write A CheckPoint at %d with avgloss %.5f" % (v_step, min_loss)
                        else:
                            print "Current Min avgloss is %.5f. Last avgloss is %.5f" % ( min_loss, avgloss)
            except Exception, e:
                coord.request_stop(e)
            finally:
                sess.run(queue.close(cancel_pending_enqueues=True))
                coord.request_stop()
                coord.join()

    def getCheckPoint(self):
        ckptfile = FLAGS.train_dir + '/checkpoint'
        f = open(ckptfile, 'rb')
        ckpt = f.readline().split(':')[1].strip().strip('"')
        f.close()
        #prefix = os.path.abspath(FLAGS.train_dir)
        #ckpt = prefix + '/' + ckpt
        return ckpt

    def top_k_error(self, predictions, labels, k):
        #batch_size = float(FLAGS.batch_size) #tf.shape(predictions)[0]
        batch_size = tf.shape(predictions)[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size - num_correct) / batch_size

def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--Continue", action='store_true', default=False,
        help="continue training from a checkpoint")
    args = parser.parse_args()
    return args.Continue

def main(argv=None):  # pylint: disable=unused-argument
    Continue = GetOptions()
    model = ResNet()
    train = Train(FLAGS.batch_size, model, FLAGS.TrainingData, FLAGS.TestingData)
    if Continue:
        ckpt = train.getCheckPoint()
        print "Train From a Check Point:", ckpt
        train.run(continueModel=ckpt)
    else:
        train.run()

if __name__ == '__main__':
    tf.app.run()
