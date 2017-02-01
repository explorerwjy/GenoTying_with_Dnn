#!/home/local/users/jw/anaconda2/bin/python
#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
#
#========================================================================================================

from optparse import OptionParser



def GetOptions():
	parser = OptionParser()
	parser.add_option('-','--',dest = '', metavar = '', help = '')
	(options,args) = parser.parse_args()
	
	return

def train():
	with tf.Graph().as_default():
		global_step = tf.contrib.framework.get_or_create_global_step()
		tensors, labels = Input.Inputs()
		logits = Models.ConvNets.Inference(tensors)
		loss = Models.ConvNets.loss(logits, labels)
		train_op = Models.ConvNets.train(loss, global_step)

		class _LoggoerHook(tf.train.SessionRunHook):
			def begin(self):
				self._step = -1

			def before_run(self, run_context):
				self._step +=1
				self._start_time = time.time()
				return tf.train.SessionRunArgs(loss)

			def after_run(self, run_context, run_values):
				duration = time().time() - self._start_time()
				loss_value = run_values.results
				if self._step % 10 == 0:
					num_examples_per_step = FLAGS.batch_size
					examples_per_sec = num_examples_per_step / duration
					sec_per_batch = float(duration)
					format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f )' 'sec/batch)')
					print format_str % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch)

		with tf.train.MonitoredTrainingSession(
				checkpoint_dir = FLAGS.train_dir,
				hooks = [tf.trainStopAtStepHook(last_step=FLAGS.amx_steps), tftrain.NanTensorHook(loss), _LoggerHook()],
				config=tf.configProto(log_device_placement=FLAGS.log_device_placement)) as mon_sess:
			while not mon_sess.should_stop():
				mon_sess.run(train_op)


def main():

	return

if __name__=='__main__':
	main()
