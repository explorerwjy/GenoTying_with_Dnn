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

GPUs = [4]
available_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([ available_devices[x] for x in GPUs])
print "Using GPU ",os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string('train_dir', './train_logs/train_0',
#                          """Directory where to checkpoint.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './train_logs/train_0',
                           """Directory where to read model checkpoints.""")

def enqueueInputData(sess, coord, Reader, enqueue_op, queue_input_data, queue_input_label, queue_input_chrom, queue_input_pos, queue_input_ref, queue_input_alt):
    try:
        while True:
            one_tensor, chrom, pos, ref, alt, label = Reader.OnceReadWithInfo()
            if one_tensor == None:
                #raise Exception('Finish Reading the file')
                return
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

    def run(self):
        s_time = time.time()
        dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
        Hand = gzip.open(self.DataFile, 'rb')
        Reader = RecordReader(Hand)
        fout = open(self.OutName, 'wb')
        with tf.Graph().as_default():
            global_step = tf.Variable(0, trainable=False, name='global_step')
            TensorPL = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, DEPTH * (HEIGHT) * WIDTH))
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
                true_count = 0  # Counts the number of correct predictions.
                total_count = 0
                print self.getCheckPoint()
                saver.restore(sess, self.getCheckPoint())
                print "CKPT starts with step",(sess.run(global_step))
                self.WriteHeadLines(fout)
                fout.write('\t'.join(["#CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT","SAMPLE"]) + '\n')
                while 1:
                    tensors, chroms, starts, refs, alts, labels = Reader.read3()
                    GL, _loss, _correct, GT = sess.run([normed_logits, loss, top_k_op, prediction], feed_dict={TensorPL: tensors, LabelPL: labels})
                    #print "loss:",_loss
                    #print "batch correct",np.sum(_correct)
                    true_count += np.sum(_correct)
                    total_count += FLAGS.batch_size
                    for chrom, start, ref, alt, label, gt, gl in zip(chroms, starts, refs, alts, labels, GT, GL):
                        self.Form_record(chrom, start, ref, alt, label, gt, gl, fout)

            except Exception, e:
                print e
                traceback.print_exc()
                fout.close()
                # Compute precision @ 1.
                print "Predicted Right:{}\t\tTotal:{}".format(true_count, total_count)
                precision = float(true_count) / total_count
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
            finally:
                pass

    def WriteHeadLines(self, fout):
        fout.write('##fileformat=VCFv4.2\n')
        fout.write('##ALT=<ID=NON_REF,Description="Represents any possible alternative allele at this location">\n')
        fout.write('##FORMAT=<ID=GL,Number=1,Type=Integer,Description="Genotype Likelihood">\n')
        fout.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        fout.write('##GenoTypingWithDeepLearning\n')
        fout.write('##INFO=<ID=Label,Number=1,Type=Integer,Description="Label of the Variant to be True or False. Used while developing and testing">\n')
        fout.write('##contig=<ID=1,assembly=hg19,length=249250621>\n') 
        fout.write('##contig=<ID=2,assembly=hg19,length=243199373>\n')
        fout.write('##contig=<ID=3,assembly=hg19,length=198022430>\n')
        fout.write('##contig=<ID=4,assembly=hg19,length=191154276>\n')
        fout.write('##contig=<ID=5,assembly=hg19,length=180915260>\n')
        fout.write('##contig=<ID=6,assembly=hg19,length=171115067>\n')
        fout.write('##contig=<ID=7,assembly=hg19,length=159138663>\n')
        fout.write('##contig=<ID=8,assembly=hg19,length=146364022>\n')
        fout.write('##contig=<ID=9,assembly=hg19,length=141213431>\n')
        fout.write('##contig=<ID=10,assembly=hg19,length=135534747>\n')
        fout.write('##contig=<ID=11,assembly=hg19,length=135006516>\n')
        fout.write('##contig=<ID=12,assembly=hg19,length=133851895>\n')
        fout.write('##contig=<ID=13,assembly=hg19,length=115169878>\n')
        fout.write('##contig=<ID=14,assembly=hg19,length=107349540>\n')
        fout.write('##contig=<ID=15,assembly=hg19,length=102531392>\n')
        fout.write('##contig=<ID=16,assembly=hg19,length=90354753>\n')
        fout.write('##contig=<ID=17,assembly=hg19,length=81195210>\n')
        fout.write('##contig=<ID=18,assembly=hg19,length=78077248>\n')
        fout.write('##contig=<ID=19,assembly=hg19,length=59128983>\n')
        fout.write('##contig=<ID=20,assembly=hg19,length=63025520>\n')
        fout.write('##contig=<ID=21,assembly=hg19,length=48129895>\n')
        fout.write('##contig=<ID=22,assembly=hg19,length=51304566>\n')
        fout.write('##contig=<ID=X,assembly=hg19,length=155270560>\n')
        fout.write('##contig=<ID=Y,assembly=hg19,length=59373566>\n')

    def Form_PhredPL(self, PL):
        if PL == 0:
            return 100
        else:
            return int(round(-10 * math.log10(PL)))

    def normPL(self, PL):
        minPL = min(PL)
        PL = map(lambda x: x-minPL, PL)
        return PL

    def Form_record(self, chrom, start, ref, alt, label, gt, gl, fout):
        if chrom == '.':
            raise Exception('Read Up All variants. Stop process')
        try:
            gl = map(lambda x: self.Form_PhredPL(x), gl)
            gl = self.normPL(gl)
            string_gl = map(str, gl)
            GL = ','.join(string_gl)
            GQ = str(sorted(gl)[1])
            if gt == 0:
                GT = '0/0'
            elif gt == 1:
                GT = '0/1'
            elif gt == 2:
                GT = '1/1'
            fout.write('\t'.join([chrom, start, ".", ref, alt, str(
                sorted(gl)[1]), ".", "Label={}".format(str(label)), "GT:GQ:PL", GT + ':' + GQ + ":" + GL]) + '\n')     
        except ValueError:
            print "Math Domain Error:", chrom, start, ref, alt, label, gt, gl

    def getCheckPoint(self):
        ckptfile = FLAGS.checkpoint_dir + '/checkpoint'
        f = open(ckptfile, 'rb')
        ckpt = f.readline().split(':')[1].strip().strip('"')
        f.close()
        prefix = os.path.abspath(FLAGS.checkpoint_dir)
        ckpt = prefix + '/' + ckpt
        return ckpt

    def run_2(self):
        s_time = time.time()
        dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
        Hand = gzip.open(self.DataFile, 'rb')
        Reader = RecordReader(Hand)
        fout = open(self.OutName, 'wb')
        with tf.Graph().as_default():
            global_step = tf.Variable(0, trainable=False, name='global_step')
            # Input Data
            queue_input_data = tf.placeholder(dtype, shape=[DEPTH * (HEIGHT) * WIDTH])
            queue_input_label = tf.placeholder(tf.int32, shape=[])
            queue_input_chrom = tf.placeholder(tf.string, shape=[])
            queue_input_pos = tf.placeholder(tf.string, shape=[])
            queue_input_ref = tf.placeholder(tf.string, shape=[])
            queue_input_alt = tf.placeholder(tf.string, shape=[])

            queue = tf.FIFOQueue(capacity=FLAGS.batch_size * 10,
                                      dtypes=[dtype, tf.int32, tf.string, tf.string, tf.string, tf.string],
                                      shapes=[[DEPTH * (HEIGHT) * WIDTH], [], [], [], [] ,[] ],
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

                    true_count += np.sum(_correct)
                    step += 1

                # Compute precision @ 1.
                print "Predicted Right:{}\t\tTotal:{}".format(true_count, total_sample_count)
                precision = float(true_count) / total_sample_count
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
                fout.close()

            except Exception, e:
                coord.request_stop(e)
            finally:
                sess.run(queue.close(cancel_pending_enqueues=True))
                coord.request_stop()
                coord.join(threads)

    def WriteBatch(self, _labels, _chrom, _pos, _ref, _alt, _PL, _GT, fout):
            for label, chrom, start, ref, alt, gl, gt in zip(_labels, _chrom, _pos, _ref, _alt, _PL, _GT):
                gl = map(lambda x: (-10 * math.log10(x), gl))
                string_gl = map(str, gl)
                GL = ','.join(string_gl)
                if gt == 0:
                    GT = '0/0'
                elif gt == 1:
                    GT = '0/1'
                elif gt == 2:
                    GT = '1/1'
                fout.write('\t'.join([chrom, start, ".", ref, alt, str(
                    min(gl)), ".", "Label={}".format(str(label)), "GT:GL", GT + ':' + GL]) + '\n')  

def main(argv=None):  # pylint: disable=unused-argument
    s_time = time.time()
    #DataFile = FLAGS.TestingData
    #DataFiles = [FLAGS.TrainingData, FLAGS.TestingData]
    DataFiles = [FLAGS.TestingData, FLAGS.TrainingData]
    for DataFile in DataFiles:
        try:
            OutName = 'Calling.' + FLAGS.checkpoint_dir.split('/')[-1]+ '.' +DataFile.strip().split('/')[-1].split('.')[0] + '.vcf'
            model = Models.ConvNets()
            caller = TensorCaller(FLAGS.batch_size, model, DataFile, OutName)
            caller.run()
        except Exception as e:
            print e
            traceback.print_exc()
    print "Total Runing Time is %.3f"%(time.time() - s_time)

if __name__ == '__main__':
    tf.app.run()
