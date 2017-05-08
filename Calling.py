#!/home/yufengshen/anaconda2/bin/python
# Author: jywang explorerwjy@gmail.com

#=========================================================================
# Calling Variants with saved model
#=========================================================================


from datetime import datetime
import math
import time
import sys
import os
import traceback
import numpy as np
import tensorflow as tf
from Input import *
import Models
from threading import Thread
sys.stdout = sys.stderr

GPUs = [6]
available_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([ available_devices[x] for x in GPUs])
print "Using GPU ",os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './train_6',
                           """Directory where to checkpoint.""")
tf.app.flags.DEFINE_integer('num_examples', 640,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './train_6',
                           """Directory where to read model checkpoints.""")

def enqueueInputData(sess, coord, Reader, enqueue_op, queue_input_data, queue_input_label, queue_input_chrom, queue_input_pos, queue_input_ref, queue_input_alt):
    try:
        while True:
            one_tensor, chrom, pos, ref, alt, label = Reader.OnceReadWithInfo()
            if one_tensor == None:
                raise Exception('Finish Reading the file')
            sess.run(
                enqueue_op,
                feed_dict={
                    queue_input_data: one_tensor,
                    queue_input_label: label,
                    queue_input_chrom: chrom,
                    queue_input_pos: pos,
                    queue_input_ref: ref,
                    queue_input_alt: alt
                    })
    except Exception as e:
        print e
        print("finished enqueueing")
        coord.request_stop(e)

class TensorCaller:
    def __init__(self, batch_size, model, DataFile, OutName):
        self.batch_size = batch_size
        print "=" * 50
        print "InputData is:", DataFile
        print "=" * 50
        self.DataFile = DataFile
        self.model = model
        self.OutName = OutName

    def run_2(self):
        s_time = time.time()
        dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
        Hand = gzip.open(self.DataFile, 'rb')
        Reader = RecordReader(Hand)
        fout = open(self.OutName, 'wb')
        with tf.Graph().as_default():
            global_step = tf.Variable(0, trainable=False, name='global_step')
            TensorPL = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, DEPTH * (HEIGHT + 1) * WIDTH))
            LabelPL = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, ))

            logits = self.model.Inference(TensorPL)
            normed_logits = tf.nn.softmax(logits, dim=-1, name=None)
            prediction = tf.argmax(normed_logits, 1)
            loss = self.model.loss(logits, LabelPL)
            #accuracy = self.model.Accuracy(logits, label_batch)
            #train_op = self.model.Train(loss, global_step)
            top_k_op = tf.nn.in_top_k(logits, LabelPL, 1)

            #summary_op = tf.summary.merge_all()
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            sess = tf.Session()
            #summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, sess.graph)
            sess.run(init)

            try:    
                num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
                true_count = 0  # Counts the number of correct predictions.
                total_sample_count = num_iter * FLAGS.batch_size
                step = 0
                print self.getCheckPoint()
                saver.restore(sess, self.getCheckPoint())
                print "CKPT starts with step",(sess.run(global_step))
                fout.write('\t'.join(["#CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT","SAMPLE"]) + '\n')
                while step < num_iter :
                    tensors, chroms, starts, refs, alts, labels = Reader.read3()
                    GL, _loss, _correct, GT = sess.run([normed_logits, loss, top_k_op, prediction], feed_dict={TensorPL: tensors, LabelPL: labels})
                    print "loss:",_loss
                    print "batch correct",np.sum(_correct)
                    true_count += np.sum(_correct)
                    step += 1
                    for chrom, start, ref, alt, label, gt, gl in zip(chroms, starts, refs, alts, labels, GT, GL):
                        self.Form_record(chrom, start, ref, alt, label, gt, gl, fout)


                    if len(chroms) < FLAGS.batch_size:
                        return

                    #_labels, _logits, _loss, predictions = sess.run([label_batch, logits, loss, top_k_op])
                    #print "labels:",_labels
                    #print "logits:",_logits
                    #print "predict", predictions

                    
                # Compute precision @ 1.
                fout.close()
                print "Predicted Right:{}\t\tTotal:{}".format(true_count, total_sample_count)
                precision = float(true_count) / total_sample_count
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
                #summary = tf.Summary()
                #summary.ParseFromString(sess.run(summary_op))
                #summary.value.add(tag='Precision @ 1', simple_value=precision)
                #summary_writer.add_summary(summary, step)
            except Exception, e:
                print e
                traceback.print_exc()
            finally:
                pass


    def Form_record(self, chrom, start, ref, alt, label, gt, gl, fout):
        string_gl = map(str, gl)
        GL = ','.join(string_gl)
        if gt == 0:
            GT = '0/0'
        elif gt == 1:
            GT = '0/1'
        elif gt == 2:
            GT = '1/1'
        fout.write('\t'.join([chrom, start, ".", ref, alt, str(
            max(gl)), ".", "Label={}".format(str(label)), "GT:GL", GT + ':' + GL]) + '\n')     

    def getCheckPoint(self):
        ckptfile = FLAGS.checkpoint_dir + '/checkpoint'
        f = open(ckptfile, 'rb')
        ckpt = f.readline().split(':')[1].strip().strip('"')
        f.close()
        prefix = os.path.abspath(FLAGS.checkpoint_dir)
        ckpt = prefix + '/' + ckpt
        return ckpt

    def run(self):
        s_time = time.time()
        dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
        Hand = gzip.open(self.DataFile, 'rb')
        Reader = RecordReader(Hand)
        fout = open(self.OutName, 'wb')
        with tf.Graph().as_default():
            global_step = tf.Variable(0, trainable=False, name='global_step')
            # Input Data
            queue_input_data = tf.placeholder(dtype, shape=[DEPTH * (HEIGHT + 1) * WIDTH])
            queue_input_label = tf.placeholder(tf.int32, shape=[])
            queue_input_chrom = tf.placeholder(tf.string, shape=[])
            queue_input_pos = tf.placeholder(tf.string, shape=[])
            queue_input_ref = tf.placeholder(tf.string, shape=[])
            queue_input_alt = tf.placeholder(tf.string, shape=[])

            queue = tf.FIFOQueue(capacity=FLAGS.batch_size * 10,
                                      dtypes=[dtype, tf.int32, tf.string, tf.string, tf.string, tf.string],
                                      shapes=[[DEPTH * (HEIGHT + 1) * WIDTH], [], [], [], [] ,[] ],
                                      name='FIFOQueue')
            enqueue_op = queue.enqueue([queue_input_data, queue_input_label, queue_input_chrom, queue_input_pos, queue_input_ref, queue_input_alt ])
            dequeue_op = queue.dequeue()
            # Get Tensors and labels for Training data.
            data_batch, label_batch, chrom_batch, pos_batch, ref_batch, alt_batch = tf.train.batch(dequeue_op, batch_size=FLAGS.batch_size, capacity=FLAGS.batch_size * 8)
            logits = self.model.Inference(data_batch)
            loss = self.model.loss(logits, label_batch)
            top_k_op = tf.nn.in_top_k(logits, label_batch, 1)

            normed_logits = tf.nn.softmax(logits, dim=-1, name=None)
            prediction = tf.argmax(normed_logits, 1)

            #summary_op = tf.summary.merge_all()
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            sess = tf.Session()
            #summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, sess.graph)
            sess.run(init)
            coord = tf.train.Coordinator()
            enqueue_thread = Thread(
                target=enqueueInputData,
                args=[sess, coord, Reader, enqueue_op, queue_input_data, queue_input_label, queue_input_chrom, queue_input_pos, queue_input_ref, queue_input_alt ])
            enqueue_thread.isDaemon()
            enqueue_thread.start()
            print "Before Threads"
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            fout.write('\t'.join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "SAMPLE"]) + '\n')
            try:    
                num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
                true_count = 0  # Counts the number of correct predictions.
                total_sample_count = num_iter * FLAGS.batch_size
                step = 0
                print self.getCheckPoint()
                saver.restore(sess, self.getCheckPoint())
                print "CKPT starts with step",(sess.run(global_step))
                while step < num_iter and not coord.should_stop():
                    _chrom, _pos, _ref, _alt = sess.run([chrom_batch, pos_batch, ref_batch, alt_batch])
                    _labels, _logits, _loss, _correct, PL, GT = sess.run([label_batch, logits, loss, top_k_op, normed_logits, prediction])
                    self.WriteBatch(_labels, _chrom, _pos, _ref, _alt, PL, GT, fout)
                    #print "labels:",_labels
                    #print "logits:",_logits
                    #print "predict", predictions
                    print "loss:",_loss
                    print "Correct One Batch", np.sum(_correct)
                    true_count += np.sum(_correct)
                    step += 1

                # Compute precision @ 1.
                print "Predicted Right:{}\t\tTotal:{}".format(true_count, total_sample_count)
                precision = float(true_count) / total_sample_count
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
                fout.close()
                #summary = tf.Summary()
                #summary.ParseFromString(sess.run(summary_op))
                #summary.value.add(tag='Precision @ 1', simple_value=precision)
                #summary_writer.add_summary(summary, step)
            except Exception, e:
                coord.request_stop(e)
            finally:
                sess.run(queue.close(cancel_pending_enqueues=True))
                coord.request_stop()
                coord.join(threads)

    def WriteBatch(self, _labels, _chrom, _pos, _ref, _alt, _PL, _GT, fout):
        for label, chrom, start, ref, alt, gl, gt in zip(_labels, _chrom, _pos, _ref, _alt, _PL, _GT):
            string_gl = map(str, gl)
            GL = ','.join(string_gl)
            if gt == 0:
                GT = '0/0'
            elif gt == 1:
                GT = '0/1'
            elif gt == 2:
                GT = '1/1'
            fout.write('\t'.join([chrom, start, ".", ref, alt, str(
                max(gl)), ".", "Label={}".format(str(label)), "GT:GL", GT + ':' + GL]) + '\n')  

def do_eval(sess, global_step, normed_logits, prediction, DataReader, tensor_pl, fout):
    counter = 0
    s_time = time.time()
    while True:
        tensor, chroms, starts, refs, alts, labels = DataReader.read3()
        GL, GT = sess.run([normed_logits, prediction],
                          feed_dict={tensor_pl: tensor})
        for chrom, start, ref, alt, label, gt, gl in zip(
                chroms, starts, refs, alts, labels, GT, GL):
            Form_record(chrom, start, ref, alt, label, gt, gl, fout)
            #gl = map(str, gl)
            # fout.write(str(gt)+'\t'+','.join(gl)+'\n')

        if len(chroms) < FLAGS.batch_size:
            return
        if counter % 10 == 0:
            duration = time.time() - s_time
            print (sess.run(global_step))
            print "Read %d batches, %d records, used %.3fs 10 batch" % (counter, counter * FLAGS.batch_size, duration)
            s_time = time.time()
        counter += 1


def Calling(Dataset, OutName, ModelCKPT):
    s_time = time.time()
    # with tf.Graph().as_default() as g:
    with tf.device('/gpu:6'):
        #TrainingData = gzip.open(TrainingData,'rb')
        Data = gzip.open(Dataset, 'rb')
        DataReader = RecordReader(Data)
        #fout_training = open('Calling_training.txt','wb')
        fout = open(OutName, 'wb')

        TensorPL = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, WIDTH * (HEIGHT + 1) * 3))

        convnets = Models.ConvNets()
        logits = convnets.Inference(TensorPL)
        normed_logits = tf.nn.softmax(logits, dim=-1, name=None)
        prediction = tf.argmax(normed_logits, 1)
        global_step = tf.Variable(0, trainable=False, name='global_step')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            sess.run(init)
            saver.restore(sess, ModelCKPT)

            print "Evaluating On", Dataset
            fout.write('\t'.join(["#CHROM",
                                  "POS",
                                  "ID",
                                  "REF",
                                  "ALT",
                                  "QUAL",
                                  "FILTER",
                                  "INFO",
                                  "FORMAT",
                                  "SAMPLE"]) + '\n')
            do_eval(
                sess,
                global_step,
                normed_logits,
                prediction,
                DataReader,
                TensorPL,
                fout)
        fout.close()
    print 'spend %.3fs on Calling %s' % (time.time() - s_time, OutName)


def Calling_2(Dataset, ModelCKPT):
    dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
    DatasetHand = gzip.open(Dataset, 'rb')
    DataReader = RecordReader(DatasetHand)

    with tf.Graph().as_default():
        queue_input_data = tf.placeholder(
            dtype, shape=[DEPTH * (HEIGHT + 1) * WIDTH])
        queue_input_label = tf.placeholder(tf.int32, shape=[])
        queue = tf.RandomShuffleQueue(capacity=FLAGS.batch_size * 10,
                                      dtypes=[dtype,
                                              tf.int32],
                                      shapes=[[DEPTH * (HEIGHT + 1) * WIDTH],
                                              []],
                                      min_after_dequeue=FLAGS.batch_size,
                                      name='RandomShuffleQueue')
        enqueue_op = queue.enqueue([queue_input_data, queue_input_label])
        dequeue_op = queue.dequeue()
        # Get Tensors and labels for Training data.
        data_batch, label_batch = tf.train.batch(
            dequeue_op, batch_size=FLAGS.batch_size, capacity=FLAGS.batch_size * 4)
        #data_batch_reshape = tf.transpose(data_batch, [0,2,3,1])

        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Build a Graph that computes the logits predictions from the
        # inference model.
        convnets = Models.ConvNets()
        logits = convnets.Inference(data_batch)

        # Calculate loss.
        loss = convnets.loss(logits, label_batch)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = convnets.Train(loss, global_step)
        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(init)

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

        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        min_loss = 100
        try:
            for step in xrange(max_steps):
                start_time = time.time()
                #_, loss_value, v_step = sess.run([train_op, loss, global_step])
                loss_value, _, v_step = sess.run([loss, train_op, global_step])
                #_, loss_value, v_step, queue_size = sess.run([train_op, loss, global_step, queue.size()])
                duration = time.time() - start_time
                if (v_step) % 100 == 0 or (v_step) == max_steps:
                    summary_str = sess.run(summary)
                    summary_writer.add_summary(summary_str, v_step)
                    summary_writer.flush()
                    # Save Model only if loss decreasing
                    if loss_value < min_loss:
                        checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                        saver.save(
                            sess, checkpoint_file, global_step=global_step)
                        min_loss = loss_value
                        print "Write A CheckPoint at %d" % (v_step)
                    #loss_value = sess.run(loss, feed_dict=feed_dict)
                    print 'Step %d Training loss = %.3f (%.3f sec); Saved loss = %.3f' % (v_step, loss_value, duration, min_loss)
                elif v_step % 10 == 0:
                    print 'Step %d Training loss = %.3f (%.3f sec)' % (v_step, loss_value, duration)
                    # print "Queue Size", queue_size
                    summary_str = sess.run(summary)
                    summary_writer.add_summary(summary_str, v_step)
                    summary_writer.flush()
        except Exception as e:
            coord.request_stop(e)
        finally:
            sess.run(queue.close(cancel_pending_enqueues=True))
            coord.request_stop()
            coord.join(threads)


def main(argv=None):  # pylint: disable=unused-argument
    DataFile = FLAGS.TrainingData
    #DataFile = FLAGS.TestingData
    try:
        OutName = 'Calling.' + DataFile.strip().split('/')[-1].split('.')[0] + '.vcf'
        model = Models.ConvNets()
        caller = TensorCaller(FLAGS.batch_size, model, DataFile, OutName)
        caller.run()
    except Exception as e:
        print e
        traceback.print_exc()

if __name__ == '__main__':
    tf.app.run()
