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


def eval_once(saver, summary_writer, top_k_op, summary_op):
    """Run Eval once.
    Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split(
                '/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                             start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
        true_count = 0  # Counts the number of correct predictions.
        total_sample_count = num_iter * FLAGS.batch_size
        step = 0
        while step < num_iter and not coord.should_stop():
            predictions = sess.run([top_k_op])
            true_count += np.sum(predictions)
            step += 1

        # Compute precision @ 1.
        precision = true_count / total_sample_count
        print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

        summary = tf.Summary()
        summary.ParseFromString(sess.run(summary_op))
        summary.value.add(tag='Precision @ 1', simple_value=precision)
        summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        #eval_data = FLAGS.eval_data == 'test'
        #images, labels = cifar10.inputs(eval_data=eval_data)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def do_eval(
        sess,
        eval_correct,
        testing_tensor_pl,
        testing_label_pl,
        dataset,
        Total):
    true_count = 0
    steps_per_epoch = Total // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    for step in xrange(steps_per_epoch):
        tensor, label = dataset.read_batch()
        feed_dict = {testing_tensor_pl: tensor, testing_label_pl: label}
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print '\tNum examples: %d\tNum correct: %d\tPrecision @ 1: %.04f' % (num_examples, true_count, precision)


def runTesting(TrainingData, ValidationData, TestingData, ModelCKPT):
    Num_training = 3522409
    Num_validation = 86504
    Num_testing = 186468

    # with tf.Graph().as_default() as g:
    with tf.device('/gpu:7'):
        TrainingData = gzip.open(TrainingData, 'rb')
        ValidationData = gzip.open(ValidationData, 'rb')
        TestingData = gzip.open(TestingData, 'rb')

        dataset_training = Window2Tensor.Data_Reader(
            TrainingData, batch_size=BATCH_SIZE)
        dataset_validation = Window2Tensor.Data_Reader(
            ValidationData, batch_size=BATCH_SIZE)
        dataset_testing = Window2Tensor.Data_Reader(
            TestingData, batch_size=BATCH_SIZE)
        TensorPL, LabelPL = Window2Tensor.placeholder_inputs(BATCH_SIZE)

        #stime = time.time()
        # print "Reading Training Dataset %d windows"%Total
        #TrainingTensor, TrainingLabel = dataset_training.read_batch()
        #tmp1time = time.time()
        # print "Finish Reading Training Dataset. %.3f"%(tmp1time-stime)
        # print "Reading Testing Dataset %d windows"%Total
        #TestingTensor, TestingLabel = dataset_testing.read_batch()
        #tmp2time = time.time()
        # print "Finish Reading Testing Dataset. %.3f"%(tmp2time-tmp1time)

        convnets = Models.ConvNets()
        # Testing on Training
        logits = convnets.Inference(TensorPL)

        correct = evaluation(logits, LabelPL)

        saver = tf.train.Saver()

        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            saver.restore(sess, ModelCKPT)

            # print TrainingLabel
            # print sess.run(logits,feed_dict = {TensorPL:TrainingTensor})

            print "Evaluating On Training Sample"
            stime = time.time()
            do_eval(
                sess,
                logits,
                TensorPL,
                LabelPL,
                dataset_training,
                Num_training)
            print "Finish Evaluating Training Dataset. %.3f" % (time.time() - stime)

            print "Evaluating On Validation Sample"
            stime = time.time()
            do_eval(
                sess,
                logits,
                TensorPL,
                LabelPL,
                dataset_validation,
                Num_validation)
            print "Finish Evaluating Validation Dataset. %.3f" % (time.time() - stime)

            print "Evaluating On Testing Sample"
            stime = time.time()
            do_eval(
                sess,
                logits,
                TensorPL,
                LabelPL,
                dataset_testing,
                Num_testing)
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
