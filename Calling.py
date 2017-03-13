#!/home/local/users/jw/anaconda2/bin/python
#Author: jywang explorerwjy@gmail.com

#========================================================================================================
# Calling Variants with saved model
#========================================================================================================


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
	ckptfile = FLAGS.checkpoint_dir+'/log/checkpoint'
	if not os.path.isfile(ckptfile):
		print "Model checkpoint not exists."
		exit()
	f = open(ckptfile,'rb')
	ckpt = f.readline().split(':')[1].strip().strip('"')
	f.close()
	prefix = os.path.abspath(FLAGS.checkpoint_dir+'/log/')
	ckpt = prefix + '/' + ckpt
	return ckpt

def Form_record(chrom, start, ref, alt, gt, gl, fout):
	string_gl = map(str, gl)
	GL = ','.join(string_gl)
	if gt == 0:
		GT = '0/0'
	elif gt == 1:
		GT = '0/1'
	elif gt == 2:
		GT = '1/1'
	fout.write('\t'.join([chrom, start, ".", ref, alt, max(gl), ".", ".", "GT:GL", GT+':'+GL]+'\n'))

# dataset: Window2Tensor.Data_Reader object, read BATCH_SIZE samples a time.
def do_eval(sess, normed_logits, prediction, DataReader, tensor_pl, fout):
	counter = 0
	s_time = time.time()
	while 1:
		tensor, chroms, starts, refs, alts = DataReader.read2()
		GL, GT = sess.run([normed_logits, prediction], feed_dict = {tensor_pl: tensor})
		for chrom, start, ref, alt, gt, gl in zip(chroms, starts, refs, alts, GT, GL):
			Form_record(chrom, start, ref, alt, gt, gl, fout)
			#gl = map(str, gl)
			#fout.write(str(gt)+'\t'+','.join(gl)+'\n')

		if len(label) < BATCH_SIZE:
			return
		if counter % 10 == 0:
			duration = time.time()-s_time()
			print "Read %d batches, %d records, used %.3fs per batch"%(counter,counter*BATCH_SIZE,duration)
			s_time = time.time()
		counter += 1

def Calling(Dataset, OutName, ModelCKPT):
	#with tf.Graph().as_default() as g:
	with tf.device('/gpu:7'):
		#TrainingData = gzip.open(TrainingData,'rb')
		Data = gzip.open(Dataset,'rb')
		DataReader = RecordReader(Data)
		#fout_training = open('Calling_training.txt','wb')
		fout = open(OutName,'wb')

		TensorPL = tf.placeholder(tf.float32, shape=(BATCH_SIZE,WIDTH*(HEIGHT+1)*3))

		convnets = Models.ConvNets()
		logits = convnets.Inference(TensorPL)
		normed_logits = tf.nn.softmax(logits, dim=-1, name=None)
		prediction=tf.argmax(normed_logits,1)

		saver = tf.train.Saver() 
		config = tf.ConfigProto(allow_soft_placement = True)
		with tf.Session(config = config) as sess:
			saver.restore(sess, ModelCKPT)
			
			print "Evaluating On",Dataset
			fout.write('\t'.join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "SAMPLE"]+'\n'))
			do_eval(sess, normed_logits, prediction, DataReader,TensorPL, fout)
		fout.close()

def Calling_2(Dataset, ModelCKPT):
	dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
    BATCH_SIZE = FLAGS.batch_size
    DatasetHand=gzip.open(Dataset,'rb')
    DataReader = RecordReader(DatasetHand)

    with tf.Graph().as_default():
        queue_input_data = tf.placeholder(dtype, shape=[DEPTH * (HEIGHT+1) * WIDTH])
        queue_input_label = tf.placeholder(tf.int32, shape=[])
        queue = tf.RandomShuffleQueue(capacity=FLAGS.batch_size*10, dtypes=[dtype, tf.int32], shapes=[[DEPTH * (HEIGHT+1) * WIDTH], []], min_after_dequeue=FLAGS.batch_size, name='RandomShuffleQueue')
        enqueue_op = queue.enqueue([queue_input_data, queue_input_label])
        dequeue_op = queue.dequeue()
        # Get Tensors and labels for Training data.
        data_batch, label_batch = tf.train.batch(dequeue_op, batch_size=FLAGS.batch_size, capacity=FLAGS.batch_size*4)
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


        enqueue_thread = Thread(target=enqueueInputData, args=[sess, coord, TrainingReader, enqueue_op, queue_input_data, queue_input_label])
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
                    #Save Model only if loss decreasing
                    if loss_value < min_loss:
                        checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_file, global_step = global_step)
                        min_loss = loss_value
                        print "Write A CheckPoint at %d"%(v_step)
                    #loss_value = sess.run(loss, feed_dict=feed_dict)
                    print 'Step %d Training loss = %.3f (%.3f sec); Saved loss = %.3f' % (v_step, loss_value, duration, min_loss)
                elif v_step % 10 == 0: 
                    print 'Step %d Training loss = %.3f (%.3f sec)' % (v_step, loss_value, duration)
                    #print "Queue Size", queue_size 
                    summary_str = sess.run(summary)
                    summary_writer.add_summary(summary_str, v_step)
                    summary_writer.flush()
        except Exception, e:
            coord.request_stop(e)
        finally:
            sess.run(queue.close(cancel_pending_enqueues=True))
            coord.request_stop()
            coord.join(threads)


def main(argv=None):  # pylint: disable=unused-argument
	if tf.gfile.Exists(FLAGS.eval_dir):
		tf.gfile.DeleteRecursively(FLAGS.eval_dir)
	tf.gfile.MakeDirs(FLAGS.eval_dir)
	
	# Get File Name of TraingData, ValidationData and Testdata
	TrainingData = FLAGS.TrainingData
	TestingData = FLAGS.TestingData
	# Get The Saved Model
	# ModelCKPT = FLAGS.checkpoint_dir+'/model.ckpt-4599.meta'

	ModelCKPT = GetCheckPoint()
	Calling(TrainingData, TestingData, ModelCKPT)



if __name__ == '__main__':
	tf.app.run()
