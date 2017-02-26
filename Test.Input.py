import tensorflow as tf
import numpy as np
import threading

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', './tmp/TensorCaller_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('train_dir', './tmp/TensorCaller_train',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('log_dir', './tmp/TensorCaller_train/log',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of WindowTensor to process in a batch.""")
tf.app.flags.DEFINE_string('TrainingData', './windows_training.txt.gz',
                           """Path to the Training Data.""")
tf.app.flags.DEFINE_string('ValidationData', './windows_validation.txt.gz',
                           """Path to the Validation Data.""")
tf.app.flags.DEFINE_string('TestingData', './windows_testing.txt.gz',
                           """Path to the Testing Data.""")
tf.app.flags.DEFINE_boolean('use_fl16', True,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('queueThreads', 8,
                            """Number of threads used to read input data.""")


class window_tensor():
    def __init__(self,line):
        self.chrom, self.start, self.end, self.label, self.window = line.strip().split('\t')
        self.Alignment = self.window[ 0 : WIDTH * (HEIGHT+1) ]
        self.Qual = self.window[ WIDTH * (HEIGHT+1) : WIDTH * (HEIGHT+1)*2]
        self.Strand = self.window[ WIDTH * (HEIGHT+1)*2 : WIDTH * (HEIGHT+1)*3]

    def encode(self):
        # This func encode,norm elements and form into tensor 
        res = [ (float(base)/6 - 0.5) for base in list(self.Alignment)] + 
              [ qual2code(x) for x in list(self.Qual)] + 
              [ float(x)/2-0.5 for x in list(self.Strand)]
		if FLAGS.use_fl16: 
			RawTensor = tf.convert_to_tensor(res, dtype=tf.float16)
		else:
        	RawTensor = tf.convert_to_tensor(res, dtype=tf.float32)
        InputTensor = tf.reshape(RawTensor, [WIDTH, HEIGHT+1, 3]) 
        return InputTensor

class RecordReader():
	def __init__(self, hand):
		self.hand = hand
	def read(self):
		record = window_tensor(self.hand.readline())
		tensor = record.encode()
		label = tf.one_hot(indices=tf.cast(record.label, tf.int16), depth=3)
		return tensor, label

def Test_Input_1():
	# TensorFlow Input Pipelines for Large Data Sets
	# Generating some simple data
	r = np.arange(0.0,100003.0)
	raw_data = np.dstack((r,r,r,r))[0]
	raw_target = np.array([[1,0,0]] * 100003)

	# are used to feed data into our queue
	queue_input_data = tf.placeholder(tf.float32, shape=[20, 4])
	queue_input_target = tf.placeholder(tf.float32, shape=[20, 3])

	queue = tf.FIFOQueue(name="InputQueue", names=['WindowTensor','Label'], capacity=FALGS.batch_size*10, dtypes=[tf.float32, tf.float32], shapes=[[WIDTH,HEIGHT,DEPTH], [NUM_CLASS]])

	enqueue_op = queue.enqueue_many([queue_input_data, queue_input_target])
	dequeue_op = queue.dequeue()

	# tensorflow recommendation:
	# capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
	#data_batch, target_batch = tf.train.batch(dequeue_op, batch_size=8, capacity=FALGS.batch_size*10)
	# use this to shuffle batches:
	data_batch, target_batch = tf.train.shuffle_batch(dequeue_op, batch_size=8, capacity=FALGS.batch_size*10, min_after_dequeue=FALGS.batch_size*3)

	def enqueue(sess):
		""" Iterates over our data puts small junks into our queue."""
		under = 0
		max = len(raw_data)
		while True:
			print("starting to write into queue")
			upper = under + 20
			print("try to enqueue ", under, " to ", upper)
			if upper <= max:
				curr_data = raw_data[under:upper]
				curr_target = raw_target[under:upper]
				under = upper
			else:
				rest = upper - max
				curr_data = np.concatenate((raw_data[under:max], raw_data[0:rest]))
				curr_target = np.concatenate((raw_target[under:max], raw_target[0:rest]))
				under = rest

				sess.run(enqueue_op, feed_dict={queue_input_data: curr_data,
					queue_input_target: curr_target})
				print("added to the queue")
				print("finished enqueueing")

	# start the threads for our FIFOQueue and batch
	sess = tf.Session()
	enqueue_thread = threading.Thread(target=enqueue, args=[sess])
	enqueue_thread.isDaemon()
	enqueue_thread.start()

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)

	# Fetch the data from the pipeline and put it where it belongs (into your model)
	for i in range(5):
		run_options = tf.RunOptions(timeout_in_ms=4000)
		curr_data_batch, curr_target_batch = sess.run([data_batch, target_batch], options=run_options)
		print(curr_data_batch)

	# shutdown everything to avoid zombies
	sess.run(queue.close(cancel_pending_enqueues=True))
	coord.request_stop()
	coord.join(threads)
	sess.close()

def TestInput_2():
	#filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])
	TrainHand=gzip.open(FLAGS.TrainingData,'rb')
	reader = RecordReader(TrainHand)
	tensor, label = reader.read()

	with tf.Session() as sess:
		# Start populating the filename queue.
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		for i in range(1200):
		# Retrieve a single instance:
			example, label = sess.run([features, col5])

		coord.request_stop()
		coord.join(threads)

def TestInput_3():
	TrainHand=gzip.open(FLAGS.TrainingData,'rb')
	reader = RecordReader(TrainHand)
	tensor, label = reader.read()

	# Create a queue, and an op that enqueues examples one at a time in the queue.
	queue = tf.RandomShuffleQueue(name="InputQueue", names=['WindowTensor','Label'], capacity=FALGS.batch_size*10, dtypes=[tf.float32, tf.float32], shapes=[[WIDTH,HEIGHT,DEPTH], [NUM_CLASS]])
	enqueue_op = queue.enqueue(tensor, label)
	return queue, enqueue_o