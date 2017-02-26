def read_window(filename_queue):
	"""Reads and parses examples from Region Window data files.
	Args:
    inputBuffer: InputBuffer form Input Tensor from.
	Returns:
    An object representing a single example, with the following fields:
      height: number of reads in the result 
      width: number of bases in the result 
      depth: number of layers in the result 
      key: a scalar string Tensor describing the chrom:pos for this example.
      label: an int32 Tensor with the label in the range 0,1,2
      uint8image: a [height, width, depth] uint8 Tensor with the read data
	"""
	"""
	class window_tensor():
		def __init__(self,line):
			self.label = line[0]
			self.Alignment = line[ 13 : 13 + WIDTH * (HEIGHT+1) ]
			self.Qual = line[ 13 + WIDTH * (HEIGHT+1) : 13 + WIDTH * (HEIGHT+1)*2]
			self.Strand = line[13 + WIDTH * (HEIGHT+1)*2 : 13 + WIDTH * (HEIGHT+1)*3]

		def encode(self):
			# This func encode elements in window tensor into tf.float32
			return map(float, list(self.Alignment)) + map(lambda x:qual2code(x), list(self.Qual)) + map(float, list(self.Strand))
	tmp = window_tensor(inputBuffer.strip())
	one_tensor = tmp.encode()
	one_label = tmp.label
	"""
	class Record(object):
		pass
	result = Record()
	
	reader = tf.TextLineReader()
	result.key, value = reader.read(filename_queue)
	
	print value	
	sess = tf.Session()
	with sess.as_default():   # or `with sess:` to close on exit
		assert sess is tf.get_default_session()
		assert value.eval() == sess.run(value)
	exit()

	# Convert from a string to a vector of uint8 that is record_bytes long.
	record_bytes = tf.decode_raw(value, tf.uint8)
	tensor_bytes = (HEIGHT) * WIDTH * 3
	# The first bytes represent the label, which we convert from uint8->int32.
	result.label = tf.cast(tf.slice(record_bytes, [0], [1]), tf.int32)
	result.pos = tf.cast(tf.slice(record_bytes, [1], [12]), tf.int32)
	# The remaining bytes after the label represent the image, which we reshape
	# from [depth * height * width] to [depth, height, width].
	depth_major = tf.reshape(
    	tf.slice(record_bytes, [13], [tensor_bytes]), [3, HEIGHT, WIDTH])
	# Convert from [depth, height, width] to [height, width, depth].
	result.tensor = tf.transpose(depth_major, [1, 2, 0])
	#print result.tensor
	return result

#  Construct a queued batch of images and labels.
def Generate_Tensor_and_label_batch(tensor, label, min_queue_examples, batch_size, shuffle):
	"""
	Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
	Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
	"""
	num_preprocess_threads = 16
	if shuffle:
		Tensors, Labels_batch = tf.train.batch([tensor, label], batch_size = batch_size, num_threads = num_preprocess_threads, capacity = min_queue_examples + 3*batch_size)
	else:
		Tensors, Labels_batch = tf.train.batch([tensor, label], batch_size = batch_size, num_threads = num_preprocess_threads, capacity = min_queue_examples + 3*batch_size)
	# How this work? Display the training Tensor as image?
	# tf.image_summary('tensors',Tensors)
	return Tensors, tf.reshape(Labels_batch, [batch_size])

def inputs(Eval, DataFile, batch_size):
	"""Construct input for CIFAR evaluation using the Reader ops.
	Args:
	DataFile: Input File contains window tensor.
    batch_size: Number of images per batch.
	Returns:
    tensors: Tensors. 4D tensor of [batch_size, WIDTH, HEIGHT, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
	"""
	if not Eval:
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
	else:
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
	# Create a queue that produces the buffer to read.
	filename_queue = tf.train.string_input_producer([DataFile])

	# Read examples from buffers in the buffer queue.
	read_input = read_window(filename_queue)
	reshaped_tensor = tf.cast(read_input.tensor, tf.float32)

	# Set the shapes of tensors.
	reshaped_tensor.set_shape([HEIGHT, WIDTH, 3])
	read_input.label.set_shape([1])

	# Ensure that the random shuffling has good mixing properties.
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

	# Generate a batch of images and labels by building up a queue of examples.
	return Generate_Tensor_and_label_batch(reshaped_tensor, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)

class Data():
	def __init__(self,TrainingHand,ValidationHand,TestingHand):
		self.TrainingHand = TrainingHand
		self.TrainingBatch = BatchSize
		self.ValidationHand = ValidationHand
		self.TestingHand = TestingHand
	def ReadingTraining(self):
		# Args:    file hand of Training windows, get [batch_size] Lines of records. 
		# Returns: 4D tensor of [batch_size, height, width, depth] size
		#		   1D tensor of [batch_size] size.
		self.Training = []
		self.TrainingLabels = []
		i = 0
		while i < self.TrainingBatch:
			l = TrainingHand.readline()
			if l == '':
				self.handle.seek(0)
				continue
			tensor, label = Record(l)
			self.Training.append(tensor)
			self.label.append(label)
			i += 1
		return self.Training, self.TrainingLabels
	def ReadingValidation(self):
		self.Validation = []
		self.ValidationLabels = []
		for l in self.ValidationHand:
			tensor, label = Record(l)
			self.Validation.append(tensor)
			self.ValidationLabels.append(labels)
		return self.Validation, self.ValidationLabels
	def ReadingTesting(self):
		self.Testing = []
		self.TestingLabels = []
		for l in self.TestingHand:
			tensor, label = Record(l)
			self.Testing.append(tensor)
			self.TestingLabels.append(labels)
		return self.Testing, self.TestingLabels
def Inputs_training(Training_fname, batch_size):
	
	
	return
def Inputs_validation_testing():

	return

def inputs(data_file):
	tensors, labels = Input.inputs(False, DataFile=data_file, batch_size=FLAGS.batch_size)
	if FLAGS.use_fl16:
		tensors = tf.cast(images, tf.float16)
		labels = tf.cast(labels, tf.float16)
	return tensors, labels



def continue_train_2(ModelCKPT):
	"""Train TensorCaller for a number of steps."""
	with tf.Graph().as_default():
		print "Locating Data File"
		TrainingData = gzip.open(FLAGS.TrainingData,'rb')
		TestingData = gzip.open(FLAGS.TestingData,'rb')
		data_sets_training = Window2Tensor.Data_Reader(TrainingData, batch_size=BATCH_SIZE)
		data_sets_testing = Window2Tensor.Data_Reader(TestingData, batch_size=BATCH_SIZE) 
		print "Training Data @%s; \nTesting Data @%s" % (os.path.abspath(FLAGS.TrainingData), os.path.abspath(FLAGS.TestingData))

		# Get Tensors and labels for Training data.
		#tensors, labels = Models.inputs(FLAGS.data_file)

		global_step = tf.Variable(0, trainable=False, name='global_step')

		# Build a Graph that computes the logits predictions from the
		# inference model.
		tensor_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)
		convnets = Models.ConvNets()
		logits = convnets.Inference(tensor_placeholder)

		# Calculate loss.
		loss = convnets.loss(logits, labels_placeholder)

		# Build a Graph that trains the model with one batch of examples and
		# updates the model parameters.
		train_op = convnets.Train(loss, global_step)
		summary = tf.summary.merge_all()

		#init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		sess = tf.Session()
		saver.restore(sess, ModelCKPT)
		#print global_step
		summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
		#sess.run(init)
		
		min_loss = 100
		for step in xrange(max_steps):
			start_time = time.time()
			feed_dict = fill_feed_dict(data_sets_training, tensor_placeholder, labels_placeholder)
			
			_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
			duration = time.time() - start_time
			v_step = sess.run(global_step)    
			if step % 10 == 0:
				print 'Step %d Training loss = %.3f (%.3f sec)' % (v_step, loss_value, duration)
				summary_str = sess.run(summary, feed_dict = feed_dict)
				summary_writer.add_summary(summary_str, step)
				summary_writer.flush()

			if (step + 1) % 100 == 0 or (step + 1) == max_steps:
				#Save Model only if loss decreasing
				#print loss_value, min_loss
				if loss_value < min_loss:
					checkpoint_file = os.path.join(log_dir, 'model.ckpt')
					saver.save(sess, checkpoint_file, global_step = global_step)
					min_loss = loss_value
				feed_dict = fill_feed_dict(data_sets_testing, tensor_placeholder, labels_placeholder)
				loss_value = sess.run(loss, feed_dict=feed_dict)
				print 'Step %d Test loss = %.3f (%.3f sec). Saved loss = %.3f' % (v_step, loss_value, duration, min_loss)

def train_2():
	"""Train TensorCaller for a number of steps."""
	with tf.Graph().as_default():
		print "Locating Data File"
		TrainingData = gzip.open(FLAGS.TrainingData,'rb')
		TestingData = gzip.open(FLAGS.TestingData,'rb')
		data_sets_training = Window2Tensor.Data_Reader(TrainingData, batch_size=BATCH_SIZE)
		data_sets_testing = Window2Tensor.Data_Reader(TestingData, batch_size=BATCH_SIZE) 
		print "Training Data @%s; \nTesting Data @%s" % (os.path.abspath(FLAGS.TrainingData), os.path.abspath(FLAGS.TestingData))
		global_step = tf.contrib.framework.get_or_create_global_step()

		# Get Tensors and labels for Training data.
		#tensors, labels = Models.inputs(FLAGS.data_file)

				
		# Build a Graph that computes the logits predictions from the
		# inference model.
		tensor_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)
		convnets = Models.ConvNets()
		logits = convnets.Inference(tensor_placeholder)

		# Calculate loss.
		loss = convnets.loss(logits, labels_placeholder)

		# Build a Graph that trains the model with one batch of examples and
		# updates the model parameters.
		train_op = convnets.Train(loss, global_step)

		class _LoggerHook(tf.train.SessionRunHook):
			"""Logs loss and runtime."""

			def begin(self):
				self._step = -1

			def before_run(self, run_context):
				self._step += 1
				self._start_time = time.time()
				return tf.train.SessionRunArgs(loss)  # Asks for loss value.

			def after_run(self, run_context, run_values):
				duration = time.time() - self._start_time
				loss_value = run_values.results
				if self._step % 100 == 0: # Output Loss Every 100 Steps Training
					num_examples_per_step = FLAGS.batch_size
					examples_per_sec = num_examples_per_step / duration
					sec_per_batch = float(duration)

					format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ' 'sec/batch)')
					print (format_str % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))
				#if self._step % 1000 == 0: # Output Loss of Evauation Data Every 100 Steps



	with tf.train.MonitoredTrainingSession(
		checkpoint_dir=FLAGS.train_dir, 
		hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps), tf.train.NanTensorHook(loss), _LoggerHook()],
		config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as mon_sess:
		while not mon_sess.should_stop():
			feed_dict = fill_feed_dict(data_sets_training, tensor_placeholder, labels_placeholder)
			#_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
			mon_sess.run(train_op, feed_dict=feed_dict)

def train_3():
	"""Train TensorCaller for a number of steps."""
	with tf.Graph().as_default():
		print "Locating Data File"
		TrainingData = gzip.open(FLAGS.TrainingData,'rb')
		TestingData = gzip.open(FLAGS.TestingData,'rb')
		data_sets_training = Window2Tensor.Data_Reader(TrainingData, batch_size=BATCH_SIZE)
		data_sets_testing = Window2Tensor.Data_Reader(TestingData, batch_size=BATCH_SIZE) 
		print "Training Data @%s; \nTesting Data @%s" % (os.path.abspath(FLAGS.TrainingData), os.path.abspath(FLAGS.TestingData))

		# Get Tensors and labels for Training data.
		#tensors, labels = Models.inputs(FLAGS.data_file)

		global_step = tf.Variable(0, trainable=False, name='global_step')

		# Build a Graph that computes the logits predictions from the
		# inference model.
		tensor_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)
		convnets = Models.ConvNets()
		logits = convnets.Inference(tensor_placeholder)

		# Calculate loss.
		loss = convnets.loss(logits, labels_placeholder)

		# Build a Graph that trains the model with one batch of examples and
		# updates the model parameters.
		train_op = convnets.Train(loss, global_step)
		summary = tf.summary.merge_all()

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()
		sess = tf.Session()
		summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
		sess.run(init)
		
		min_loss = 100
		for step in xrange(max_steps):
			start_time = time.time()
			feed_dict = fill_feed_dict(data_sets_training, tensor_placeholder, labels_placeholder)
			
			_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
			duration = time.time() - start_time
			v_step = sess.run(global_step)
			if step % 10 == 0:
				print 'Step %d Training loss = %.3f (%.3f sec)' % (v_step, loss_value, duration)
				summary_str = sess.run(summary, feed_dict = feed_dict)
				summary_writer.add_summary(summary_str, v_step)
				summary_writer.flush()

			if (step + 1) % 100 == 0 or (step + 1) == max_steps:
				#Save Model only if loss decreasing
				if loss_value < min_loss:
					checkpoint_file = os.path.join(log_dir, 'model.ckpt')
					saver.save(sess, checkpoint_file, global_step = global_step)
					min_loss = loss_value
				feed_dict = fill_feed_dict(data_sets_testing, tensor_placeholder, labels_placeholder)
				loss_value = sess.run(loss, feed_dict=feed_dict)
				print 'Step %d Test loss = %.3f (%.3f sec); Saved loss = %.3f' % (v_step, loss_value, duration, min_loss)




				