#!/home/yufengshen/anaconda2/bin/python
# Author: jywang explorerwjy@gmail.com

#=========================================================================
# Training The ResNet for Tensor Caller
#=========================================================================

from ResNet import * 
import tensorflow as tf

MOMENTUM = 0.9

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '.training_logs/resnet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_float('learning_rate', 1e-4, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 1e8, "max steps")
tf.app.flags.DEFINE_boolean('resume', False,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')


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
            logits = self.model.Inference(data_batch)
            loss = self.model.loss(logits, label_batch)
            #accuracy = self.model.Accuracy(logits, label_batch)
            train_op = self.model.Train(loss, global_step, init_lr, optimizer)
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

                    if v_step % 10 == 0:
                        loss_queue.enqueue(loss_value)
                        #avgloss = loss_queue.avgloss()
                        num_examples_per_step = FLAGS.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration / FLAGS.num_gpus
                        format_str = ('%s: step %d, loss = %.5f (%.1f examples/sec; %.3f '
                          'sec/batch)')
                        print (format_str % (datetime.now(), v_step, loss_value,
                                 examples_per_sec, sec_per_batch))
                    
                    if v_step % 100 == 0:
                        prediction = float((np.sum(sess.run(top_k_op))))
                        print '@ Step {}: \t {} in {} Correct, Batch precision @ 1 ={}'.format(v_step, prediction, self.batch_size, prediction/self.batch_size)
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
        prefix = os.path.abspath(FLAGS.train_dir)
        ckpt = prefix + '/' + ckpt
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
