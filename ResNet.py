import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import datetime
import numpy as np
import os
import time
from Input import *

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]

WEIGHT_DECAY = 1
WEIGHT_DECAY_2 = 1
Keep_Prop = 0.5

FLAGS = tf.app.flags.FLAGS

class Config:
    def __init__(self):
        root = self.Scope('')
        for k, v in FLAGS.__dict__['__flags'].iteritems():
            root[k] = v
        self.stack = [ root ]

    def iteritems(self):
        return self.to_dict().iteritems()

    def to_dict(self):
        self._pop_stale()
        out = {}
        # Work backwards from the flags to top fo the stack
        # overwriting keys that were found earlier.
        for i in range(len(self.stack)):
            cs = self.stack[-i]
            for name in cs:
                out[name] = cs[name]
        return out

    def _pop_stale(self):
        var_scope_name = tf.get_variable_scope().name
        top = self.stack[0]
        while not top.contains(var_scope_name):
            # We aren't in this scope anymore
            self.stack.pop(0)
            top = self.stack[0]

    def __getitem__(self, name):
        self._pop_stale()
        # Recursively extract value
        for i in range(len(self.stack)):
            cs = self.stack[i]
            if name in cs:
                return cs[name]

        raise KeyError(name)

    def set_default(self, name, value):
        if not name in self:
            self[name] = value

    def __contains__(self, name):
        self._pop_stale()
        for i in range(len(self.stack)):
            cs = self.stack[i]
            if name in cs:
                return True
        return False

    def __setitem__(self, name, value):
        self._pop_stale()
        top = self.stack[0]
        var_scope_name = tf.get_variable_scope().name
        assert top.contains(var_scope_name)

        if top.name != var_scope_name:
            top = self.Scope(var_scope_name)
            self.stack.insert(0, top)

        top[name] = value

    class Scope(dict):
        def __init__(self, name):
            self.name = name

        def contains(self, var_scope_name):
            return var_scope_name.startswith(self.name)

class ResNet():
    def __init__(self):
        self.activation = tf.nn.relu

    def Inference(self, x, is_training=True, num_class=3, num_blocks=[3, 4, 6, 3], use_bias=False, bottleneck=True):
        x = tf.reshape(x, [-1, DEPTH, HEIGHT, WIDTH])
        x = tf.transpose(x, [0, 2, 3, 1])

        c = Config()
        c["bottleneck"] = bottleneck
        c["is_training"] = tf.convert_to_tensor(is_training, dtype="bool", name="is_training")
        c["ksize"] = 3
        c["stride"] = 1
        c["use_bias"] = use_bias
        c["fc_units_out"] = num_class
        c["num_blocks"] = num_blocks
        c["stack_stride"] = 2
        print x
        with tf.variable_scope("scale1"):
            c["conv_filters_out"] = 64
            c["ksize"] = 7
            c["stride"] = 2
            x = self.conv(x, c)
            x = self.bn(x, c)
            x = self.activation(x)
        print x
        with tf.variable_scope("scale2"):
            x = self._max_pool(x, ksize=3, stride=2)
            c["num_blocks"] = num_blocks[0]
            c["stack_stride"] = 1
            c["block_filters_internal"] = 64
            x = self.stack(x, c)

        with tf.variable_scope("scale3"):
            #x = _max_pool(x, ksize=3, stride=2)
            c["num_blocks"] = num_blocks[1]
            #c["stack_stride"] = 1
            c["block_filters_internal"] = 128
            assert c["stack_stride"] == 2
            x = self.stack(x, c)

        with tf.variable_scope("scale4"):
            #x = _max_pool(x, ksize=3, stride=2)
            c["num_blocks"] = num_blocks[2]
            #c["stack_stride"] = 1
            c["block_filters_internal"] = 256
            x = self.stack(x, c)

        with tf.variable_scope("scale5"):
            #x = _max_pool(x, ksize=3, stride=2)
            c["num_blocks"] = num_blocks[3]
            #c["stack_stride"] = 1
            c["block_filters_internal"] = 512
            x = self.stack(x, c)

        x = tf.reduce_mean(x, reduction_indices=[1,2], name="avg_pool")

        if num_class != None:
            with tf.variable_scope("fc"):
                x = self.fc(x, c)

        return x

    def stack(self, x, c):
        for n in range(c["num_blocks"]):
            s = c["stack_stride"] if n == 0 else 1
            c["block_stride"] = s
            with tf.variable_scope("block%d"%(n+1)):
                x = self.block(x, c)
            return x

    def block(self, x, c):
        filters_in = x.get_shape()[-1]
        m = 4 if c["bottleneck"] else 1
        filters_out = m * c["block_filters_internal"]
        shortcut = x # branch 1
        c["conv_filters_out"] = c["block_filters_internal"]
        if c["bottleneck"]:
            with tf.variable_scope("a"):
                c["ksize"] = 1
                c["stride"] = c["block_stride"]
                x = self.conv(x, c)
                x = self.bn(x, c)
                x = self.activation(x)
            with tf.variable_scope("b"):
                x = self.conv(x, c)
                x = self.bn(x, c)
                x = self.activation(x)
            with tf.variable_scope("c"):
                c["conv_filters_out"] = filters_out
                c["ksize"] = 1
                assert c["stride"] == 1
                x = self.conv(x, c)
                x = self.bn(x, c)

        else:
            with tf.variable_scope("A"):
                c["stride"] = c["block_stride"]
                assert c["ksize"] == 3
                x = self.conv(x, c)
                x = self.bn(x, c)
                x = self.activation(x)
            with tf.variable_scope("B"):
                c["conv_filters_out"] = filters_out
                assert c["ksize"] == 3
                assert c["stride"] == 1
                x = self.conv(x, c)
                x = self.bn(x, c)

        with tf.variable_scope("shortcut"):
            if filters_out != filters_in or c["block_stride"] != 1:
                c["ksize"] = 1
                c["stride"] = c["block_stride"]
                c["conv_filters_out"] = filters_out
                shortcut = self.conv(shortcut, c)
                shortcut = self.bn(shortcut, c)

            return self.activation(x + shortcut)

    def conv(self, x, c):
        ksize = c["ksize"]
        stride = c["stride"]
        filters_out = c["conv_filters_out"]
        filters_in = x.get_shape()[-1] #Depth
        shape = [ksize, ksize, filters_in, filters_out]
        initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
        weights = self._get_variable("weights", shape=shape, dtype="float", initializer=initializer, weight_decay=CONV_WEIGHT_DECAY)
        return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')

    def bn(self, x, c):
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]
        if c["use_bias"]:
            bias = self._get_variable("bias", params_shape, initializer=tf.zeros_initializer)
            return x + bias
        axis = list(range(len(x_shape) - 1))
        beta = self._get_variable("beta", params_shape, initializer=tf.zeros_initializer)
        gamma = self._get_variable("gamma", params_shape, initializer=tf.zeros_initializer)
        moving_mean = self._get_variable("moving_mean", params_shape, initializer=tf.zeros_initializer, trainable=False)
        moving_variance = self._get_variable("moving_variance", params_shape, initializer=tf.zeros_initializer, trainable=False)
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
        mean, variance = control_flow_ops.cond(c["is_training"], lambda: (mean, variance), lambda: (moving_mean, moving_variance))
        x =tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
        return x 

    def _max_pool(self, x, ksize=3, stride=2):
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding="SAME")

    def fc(self, x, c):
        num_units_in = x.get_shape()[-1]
        num_units_out = c["fc_units_out"]
        weights_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)
        weights = self._get_variable("weights", shape=[num_units_in, num_units_out], initializer=weights_initializer, weight_decay=FC_WEIGHT_DECAY)
        biases = self._get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer)
        x = tf.nn.xw_plus_b(x, weights, biases)
        return x

    def _get_variable(self, name, shape, initializer, weight_decay=0.0, dtype="float32", trainable=True):
        if weight_decay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None
        collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
        return tf.get_variable(name, shape, initializer=initializer, dtype=dtype, regularizer=regularizer, collections=collections, trainable=trainable)

    def loss(self, logits, labels):
        #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
        tf.summary.scalar('loss', loss_)
        return loss_

    def add_loss_summaries(self, total_loss):
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        for l in losses + [total_loss]:
            tf.summary.scalar(l.op.name + ' (raw) ', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))
        return loss_averages_op

    def Train(self, total_loss, global_step, init_lr, optimizer='Adam'):
        #num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        #decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        decay_steps = LEARNING_RATE_DECAY_STEP
        #lr = tf.train.exponential_decay(init_lr , global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)
        lr = tf.constant(init_lr)
        tf.summary.scalar('learning_rate', lr)
        loss_averages_op = self.add_loss_summaries(total_loss)

        with tf.control_dependencies([loss_averages_op]):
            #opt = tf.train.GradientDescentOptimizer(lr)
            if optimizer == 'RMSProp':
                opt = tf.train.RMSPropOptimizer(lr, decay=0.9, momentum=0.8, epsilon=1e-10, centered=False)
            elif optimizer == 'Adam':
                opt = tf.train.AdamOptimizer(lr)
            else:
                print "No Such Optimizer"
                exit()
            grads = opt.compute_gradients(total_loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(
            tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
            train_op = tf.no_op(name='train')
        return train_op

def _variable_on_cpu(name, shape, initializer):
    dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
    var = tf.get_variable(
            name,
            shape,
            initializer=initializer,
            dtype=dtype)
    return var

def _activation_summary(x):
    TOWER_NAME = 'Tower'
    tensor_name = re.sub('%s_[0-9]/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_with_weight_decay(name, shape, stddev, wd):
    dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
    var = _variable_on_cpu(
        name, shape, tf.truncated_normal_initializer(
            stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var
