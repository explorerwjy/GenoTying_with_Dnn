#!/home/local/users/jw/anaconda2/bin/python
# Author: jywang	explorerwjy@gmail.com

#=========================================================================
# Models would like to try on Tensor Variant Caller
#=========================================================================

from optparse import OptionParser
import tensorflow as tf
import re
from Input import *

Keep_Prop = 0.5
WEIGHT_DECAY = 0
WEIGHT_DECAY_2 = 4e-5


class Models():
    def Inference_1(self, RawTensor):
        print RawTensor
        InputTensor = tf.reshape(RawTensor, [-1, WIDTH, HEIGHT, 3])
        print InputTensor
        # conv1
        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 3, 32], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                InputTensor, kernel, [
                    1, 2, 2, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [32], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv1)
        # pool1
        #pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool1')
        # norm1
        #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')
        print conv1
        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 32, 64], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [64], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv2)
        print conv2
        # pool1
        pool1 = tf.nn.max_pool(
            conv2, ksize=[
                1, 3, 3, 1], strides=[
                1, 2, 2, 1], padding='SAME', name='pool1')
        print pool1
        # conv3
        with tf.variable_scope('conv3') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [128], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv3)
        print conv3
        # conv4
        with tf.variable_scope('conv4') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [128], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv4)
        print conv4

        # conv5
        with tf.variable_scope('conv5') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[5, 5, 128, 256], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [256], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv5 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv5)
        print conv5
        # norm2
        # norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')
        # pool2
        pool2 = tf.nn.max_pool(
            conv5, ksize=[
                1, 3, 3, 1], strides=[
                1, 2, 2, 1], padding='SAME', name='pool2')
        print pool2
        # local6
        with tf.variable_scope('local6') as scope:
            reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = _variable_with_weight_decay(
                'weights', shape=[dim, 384], stddev=0.04, wd=0.004)
            biases = _variable_on_cpu(
                'biases', [384], tf.constant_initializer(0.1))
            local6 = tf.nn.relu(
                tf.matmul(
                    reshape,
                    weights) + biases,
                name=scope.name)
            local6_drop = tf.nn.dropout(local6, 0.9)
            _activation_summary(local6_drop)
        print local6_drop
        # local7
        with tf.variable_scope('local7') as scope:
            weights = _variable_with_weight_decay(
                'weights', shape=[384, 192], stddev=0.04, wd=0.004)
            biases = _variable_on_cpu(
                'biases', [192], tf.constant_initializer(0.1))
            local7 = tf.nn.relu(
                tf.matmul(
                    local6_drop,
                    weights) + biases,
                name=scope.name)
            local7_drop = tf.nn.dropout(local7, 0.9)
            _activation_summary(local7_drop)
        print local7_drop
        # linear layer (WX + b)
        with tf.variable_scope('softmax') as scope:
            weights = _variable_with_weight_decay(
                'weights', [192, NUM_CLASSES], stddev=1 / 192.0, wd=0.0)
            biases = _variable_on_cpu(
                'biases', [NUM_CLASSES], tf.constant_initializer(0.0))
            softmax_linear = tf.add(
                tf.matmul(
                    local7_drop,
                    weights),
                biases,
                name=scope.name)
            _activation_summary(softmax_linear)
            #softmax = tf.nn.softmax(softmax_linear, dim=-1, name=None)
        print softmax_linear
        return softmax_linear
    
    def Inference_2(self, RawTensor):
        print RawTensor
        InputTensor = tf.reshape(RawTensor, [-1, WIDTH, HEIGHT, 3])
        print InputTensor
        # conv1
        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 3, 32], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                InputTensor, kernel, [
                    1, 2, 2, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [32], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv1)
        # pool1
        #pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool1')
        # norm1
        #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')
        print conv1
        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 32, 64], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [64], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv2)
        print conv2
        # pool1
        pool1 = tf.nn.max_pool(
            conv2, ksize=[
                1, 3, 3, 1], strides=[
                1, 2, 2, 1], padding='SAME', name='pool1')
        print pool1
        # conv3
        with tf.variable_scope('conv3') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [128], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv3)
        print conv3
        # conv4
        with tf.variable_scope('conv4') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [128], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv4)
        print conv4

        # conv5
        with tf.variable_scope('conv5') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[5, 5, 128, 256], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [256], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv5 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv5)
        print conv5
        # norm2
        # norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')
        # pool2
        pool2 = tf.nn.max_pool(
            conv5, ksize=[
                1, 3, 3, 1], strides=[
                1, 2, 2, 1], padding='SAME', name='pool2')
        print pool2
        # local6
        with tf.variable_scope('local6') as scope:
            reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = _variable_with_weight_decay(
                'weights', shape=[dim, 384], stddev=0.04, wd=0.004)
            biases = _variable_on_cpu(
                'biases', [384], tf.constant_initializer(0.1))
            local6 = tf.nn.relu(
                tf.matmul(
                    reshape,
                    weights) + biases,
                name=scope.name)
            local6_drop = tf.nn.dropout(local6, 0.9)
            _activation_summary(local6_drop)
        print local6_drop
        # local7
        with tf.variable_scope('local7') as scope:
            weights = _variable_with_weight_decay(
                'weights', shape=[384, 192], stddev=0.04, wd=0.004)
            biases = _variable_on_cpu(
                'biases', [192], tf.constant_initializer(0.1))
            local7 = tf.nn.relu(
                tf.matmul(
                    local6_drop,
                    weights) + biases,
                name=scope.name)
            local7_drop = tf.nn.dropout(local7, 0.9)
            _activation_summary(local7_drop)
        print local7_drop
        # linear layer (WX + b)
        with tf.variable_scope('softmax') as scope:
            weights = _variable_with_weight_decay(
                'weights', [192, NUM_CLASSES], stddev=1 / 192.0, wd=0.0)
            biases = _variable_on_cpu(
                'biases', [NUM_CLASSES], tf.constant_initializer(0.0))
            softmax_linear = tf.add(
                tf.matmul(
                    local7_drop,
                    weights),
                biases,
                name=scope.name)
            _activation_summary(softmax_linear)
            #softmax = tf.nn.softmax(softmax_linear, dim=-1, name=None)
        print softmax_linear
        return softmax_linear

    def VGGv1(self, RawTensor):
        print RawTensor
        #InputTensor = tf.reshape(RawTensor, [-1, WIDTH, HEIGHT, 3])
        Depth_major = tf.reshape(RawTensor, [-1, DEPTH, HEIGHT, WIDTH])
        print Depth_major
        InputTensor = tf.transpose(Depth_major, [0, 2, 3, 1])
        print InputTensor
        # ==========================================================================================
        # conv1 3-64
        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 3, 64], stddev=5e-2, wd=WEIGHT_DECAY)
            conv = tf.nn.conv2d(
                InputTensor, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [64], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv1)
            print conv1
        # ==========================================================================================
        # ==========================================================================================
        # MaxPooling
        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
        # ==========================================================================================
        # ==========================================================================================
        # conv3 3-128
        with tf.variable_scope('conv3') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 64, 64], stddev=5e-2, wd=WEIGHT_DECAY)
            conv = tf.nn.conv2d(
                pool1, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [64], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv3)
            print conv3
        # ==========================================================================================
        # ==========================================================================================
        # MaxPooling
        pool2 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
        # ==========================================================================================
        # ==========================================================================================
        # conv5 3-256
        with tf.variable_scope('conv5') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=WEIGHT_DECAY)
            conv = tf.nn.conv2d(
                pool2, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [128], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv5 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv5)
            print conv5
        # ==========================================================================================
        # ==========================================================================================
        # conv6 3-256
        with tf.variable_scope('conv6') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=WEIGHT_DECAY)
            conv = tf.nn.conv2d(
                conv5, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [128], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv6 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv6)
            print conv6
        # ==========================================================================================
        # ==========================================================================================
        # MaxPooling
        pool3 = tf.nn.max_pool(conv6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool3')
        # ==========================================================================================
        # ==========================================================================================
        # conv9 3-512
        with tf.variable_scope('conv9') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 128, 256], stddev=5e-2, wd=WEIGHT_DECAY)
            conv = tf.nn.conv2d(
                pool3, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [256], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv9 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv9)
            print conv9
        # ==========================================================================================
        # ==========================================================================================
        # conv10 3-512
        with tf.variable_scope('conv10') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 256, 256], stddev=5e-2, wd=WEIGHT_DECAY)
            conv = tf.nn.conv2d(
                conv9, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [256], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv10 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv10)
            print conv10
        # ==========================================================================================
        # ==========================================================================================
        # MaxPooling
        pool4 = tf.nn.max_pool(conv10, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool4')
        # ==========================================================================================
        # ==========================================================================================
        # conv13 3-512
        with tf.variable_scope('conv13') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 256, 512], stddev=5e-2, wd=WEIGHT_DECAY)
            conv = tf.nn.conv2d(
                pool4, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv13 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv13)
            print conv13
        # ==========================================================================================
        # ==========================================================================================
        # conv14 3-512
        with tf.variable_scope('conv14') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=WEIGHT_DECAY)
            conv = tf.nn.conv2d(
                conv13, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv14 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv14)
            print conv14
        # ==========================================================================================
        # ==========================================================================================
        # MaxPooling
        pool5 = tf.nn.max_pool(conv14, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool5')
        # ==========================================================================================
        # ==========================================================================================
        # local1
        with tf.variable_scope('local1') as scope:
            reshape = tf.reshape(pool5, [FLAGS.batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = _variable_with_weight_decay(
                'weights', shape=[dim, 4096], stddev=0.04, wd=WEIGHT_DECAY_2)
            biases = _variable_on_cpu(
                'biases', [4096], tf.constant_initializer(0.1))
            local1 = tf.nn.relu(
                tf.matmul(
                    reshape,
                    weights) + biases,
                name=scope.name)
            _activation_summary(local1)
            local1_drop = tf.nn.dropout(local1, Keep_Prop)
            _activation_summary(local1_drop)
        print local1_drop
        # ==========================================================================================
        # ==========================================================================================
        # local2
        with tf.variable_scope('local2') as scope:
            weights = _variable_with_weight_decay(
                'weights', shape=[4096, 4096], stddev=0.04, wd=WEIGHT_DECAY_2)
            biases = _variable_on_cpu(
                'biases', [4096], tf.constant_initializer(0.1))
            local2 = tf.nn.relu(
                tf.matmul(
                    local1_drop,
                    weights) + biases,
                name=scope.name)
            local2_drop = tf.nn.dropout(local2, Keep_Prop)
            _activation_summary(local2)
        print local2
        # ==========================================================================================
        # ==========================================================================================
        # local3
        with tf.variable_scope('local3') as scope:
            weights = _variable_with_weight_decay(
                'weights', shape=[4096, 1000], stddev=0.04, wd=0)
            biases = _variable_on_cpu(
                'biases', [1000], tf.constant_initializer(0.1))
            local3 = tf.nn.relu(
                tf.matmul(
                    local2_drop,
                    weights) + biases,
                name=scope.name)
            #local7_drop = tf.nn.dropout(local3, 0.5)
            _activation_summary(local3)
        print local3
        # ==========================================================================================
        # ==========================================================================================
        # linear layer (WX + b)
        with tf.variable_scope('softmax') as scope:
            weights = _variable_with_weight_decay(
                'weights', [1000, NUM_CLASSES], stddev=1 / 1000.0, wd=0.0)
            biases = _variable_on_cpu(
                'biases', [NUM_CLASSES], tf.constant_initializer(0.0))
            softmax_linear = tf.add(
                tf.matmul(
                    local3,
                    weights),
                biases,
                name=scope.name)
            _activation_summary(softmax_linear)
            #softmax = tf.nn.softmax(softmax_linear, dim=-1, name=None)
        print softmax_linear
        # ==========================================================================================
        return softmax_linear

    def VGGv3(self, RawTensor):
        print RawTensor
        InputTensor = tf.reshape(RawTensor, [-1, WIDTH, HEIGHT, 3])
        print InputTensor
        # ==========================================================================================
        # conv1
        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 3, 64], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                InputTensor, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [64], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv1)
            print conv1
        # ==========================================================================================
        # ==========================================================================================
        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 64, 64], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv1, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [64], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv2)
            print conv2
        # ==========================================================================================
        # ==========================================================================================
        # MaxPooling
        pool1 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
        # ==========================================================================================
        # ==========================================================================================
        # conv3
        with tf.variable_scope('conv3') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                pool1, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [128], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv3)
            print conv3
        # ==========================================================================================
        # ==========================================================================================
        # conv4
        with tf.variable_scope('conv4') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv3, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [128], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv4)
            print conv4
        # ==========================================================================================
        # ==========================================================================================
        # MaxPooling
        pool2 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
        # ==========================================================================================
        # ==========================================================================================
        # conv5
        with tf.variable_scope('conv5') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 128, 256], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                pool2, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [256], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv5 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv5)
            print conv5
        # ==========================================================================================
        # ==========================================================================================
        # conv6
        with tf.variable_scope('conv6') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 256, 256], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv5, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [256], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv6 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv6)
            print conv6
        # ==========================================================================================
        # ==========================================================================================
        # conv7
        with tf.variable_scope('conv7') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 256, 256], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv6, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [256], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv7 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv7)
            print conv7
        # ==========================================================================================
        # ==========================================================================================
        # conv8
        with tf.variable_scope('conv8') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 256, 256], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv7, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [256], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv8 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv8)
            print conv8
        # ==========================================================================================
        # ==========================================================================================
        # MaxPooling
        pool3 = tf.nn.max_pool(conv8, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool3')
        # ==========================================================================================
        # ==========================================================================================
        # conv9
        with tf.variable_scope('conv9') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 256, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                pool3, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv9 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv9)
            print conv9
        # ==========================================================================================
        # ==========================================================================================
        # conv10
        with tf.variable_scope('conv10') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv9, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv10 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv10)
            print conv10
        # ==========================================================================================
        # ==========================================================================================
        # conv11
        with tf.variable_scope('conv11') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv10, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv11 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv11)
            print conv11
        # ==========================================================================================
        # ==========================================================================================
        # conv12
        with tf.variable_scope('conv12') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv11, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv12 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv12)
            print conv12
        # ==========================================================================================
        # ==========================================================================================
        # MaxPooling
        pool4 = tf.nn.max_pool(conv12, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool4')
        # ==========================================================================================
        # ==========================================================================================
        # conv13
        with tf.variable_scope('conv13') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                pool4, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv13 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv13)
            print conv13
        # ==========================================================================================
        # ==========================================================================================
        # conv14
        with tf.variable_scope('conv14') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv13, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv14 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv14)
            print conv14
        # ==========================================================================================
        # ==========================================================================================
        # conv15
        with tf.variable_scope('conv15') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv14, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv15 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv15)
            print conv15
        # ==========================================================================================
        # ==========================================================================================
        # conv16
        with tf.variable_scope('conv16') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv15, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv16 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv16)
            print conv16
        # ==========================================================================================
        # ==========================================================================================
        # MaxPooling
        pool5 = tf.nn.max_pool(conv16, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool5')
        # ==========================================================================================
        # ==========================================================================================
        # local1
        with tf.variable_scope('local1') as scope:
            reshape = tf.reshape(pool5, [FLAGS.batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = _variable_with_weight_decay(
                'weights', shape=[dim, 4096], stddev=0.04, wd=0.004)
            biases = _variable_on_cpu(
                'biases', [4096], tf.constant_initializer(0.1))
            local1 = tf.nn.relu(
                tf.matmul(
                    reshape,
                    weights) + biases,
                name=scope.name)
            _activation_summary(local1)
            #local1_drop = tf.nn.dropout(local1, 0.9)
            #_activation_summary(local6_drop)
        print local1
        # ==========================================================================================
        # ==========================================================================================
        # local2
        with tf.variable_scope('local2') as scope:
            weights = _variable_with_weight_decay(
                'weights', shape=[4096, 4096], stddev=0.04, wd=0.004)
            biases = _variable_on_cpu(
                'biases', [4096], tf.constant_initializer(0.1))
            local2 = tf.nn.relu(
                tf.matmul(
                    local1,
                    weights) + biases,
                name=scope.name)
            #local7_drop = tf.nn.dropout(local2, 0.9)
            _activation_summary(local2)
        print local2
        # ==========================================================================================
        # ==========================================================================================
        # local3
        with tf.variable_scope('local3') as scope:
            weights = _variable_with_weight_decay(
                'weights', shape=[4096, 1000], stddev=0.04, wd=0.004)
            biases = _variable_on_cpu(
                'biases', [1000], tf.constant_initializer(0.1))
            local3 = tf.nn.relu(
                tf.matmul(
                    local2,
                    weights) + biases,
                name=scope.name)
            #local7_drop = tf.nn.dropout(local3, 0.9)
            _activation_summary(local3)
        print local3
        # ==========================================================================================
        # ==========================================================================================
        # linear layer (WX + b)
        with tf.variable_scope('softmax') as scope:
            weights = _variable_with_weight_decay(
                'weights', [1000, NUM_CLASSES], stddev=1 / 1000.0, wd=0.0)
            biases = _variable_on_cpu(
                'biases', [NUM_CLASSES], tf.constant_initializer(0.0))
            softmax_linear = tf.add(
                tf.matmul(
                    local3,
                    weights),
                biases,
                name=scope.name)
            _activation_summary(softmax_linear)
            #softmax = tf.nn.softmax(softmax_linear, dim=-1, name=None)
        print softmax_linear
        # ==========================================================================================
        return softmax_linear

class ResNet():
    def __init__(self):
        self.activation = tf.nn.relu
    def inference(x, is_training, num_class=3, num_blocks=[3, 4, 6, 3], use_bias=False, bottleneck=True):
        c = Config()
        c["bottleneck"] = bottleneck
        c["is_training"] = tf.convert_to_tensor(is_training, dtype="bool", name="is_training")
        c["ksize"] = 3
        c["stride"] = 1
        c["use_bias"] = use_bias
        c["fc_units_out"] = num_class
        c["num_blocks"] = num_blocks
        c["stack_stride"] = 2

        with tf.variable_scope("scale1"):
            c["conv_filters_out"] = 64
            c["ksize"] = 7
            c["stride"] = 2
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope("scale2"):
            x = _max_pool(x, ksize=3, stride=2)
            c["num_blocks"] = num_blocks[0]
            c["stack_stride"] = 1
            c["block_filters_internal"] = 64
            x = stack(x, c)

        with tf.variable_scope("scale3"):
            #x = _max_pool(x, ksize=3, stride=2)
            c["num_blocks"] = num_blocks[1]
            #c["stack_stride"] = 1
            c["block_filters_internal"] = 128
            assert c["stack_stride"] == 2
            x = stack(x, c)

        with tf.variable_scope("scale4"):
            #x = _max_pool(x, ksize=3, stride=2)
            c["num_blocks"] = num_blocks[2]
            #c["stack_stride"] = 1
            c["block_filters_internal"] = 256
            x = stack(x, c)

        with tf.variable_scope("scale5"):
            #x = _max_pool(x, ksize=3, stride=2)
            c["num_blocks"] = num_blocks[3]
            #c["stack_stride"] = 1
            c["block_filters_internal"] = 512
            x = stack(x, c)

        x = tf.reduce_mean(x, reduction_indices=[1,2], name="avg_pool")

        if num_class != None:
            with tf.variable_scope("fc"):
                x = fc(x, c)

        return x

    def stack(x, c):
        for n in range(c["num_blocks"]):
            s = c["stack_stride"] if n == 0 else 1
            c["block_stride"] = s
            with tf.variable_scope("block%d"%(n+1)):
                x = block(x, c)
            return x

    def block(x, c):
        filters_in = x.get_shape()[-1]
        m = 4 if c["bottleneck"] else 1
        filters_out = m * c["block_filters_internal"]
        shortcut = x # branch 1
        c["conv_filters_out"] = c["block_filters_internal"]
        if c["bottleneck"]:
            with tf.variable_scope("a"):
                c["ksize"] = 1
                c["stride"] = c["block_stride"]
                x = conv(x, c)
                x = bn(x, c)
                x = activation(x)
            with tf.variable_scope("b"):
                x = conv(x, c)
                x = bn(x, c)
                x = activation(x)
            with tf.variable_scope("c"):
                c["conv_filters_out"] = filters_out
                c["ksize"] = 1
                assert c["stride"] == 1
                x = conv(x, c)
                x = bn(x, c)

        else:
            with tf.variable_scope("A"):
                c["stride"] = c["block_stride"]
                assert c["ksize"] == 3
                x = conv(x, c)
                x = bn(x, c)
                x = activation(x)
            with tf.variable_scope("B"):
                c["conv_filters_out"] = filters_out
                assert c["ksize"] == 3
                assert c["stride"] == 1
                x = conv(x, c)
                x = bn(x, c)

        with tf.variable_scope("shortcut"):
            if filters_out != filters_int or c["block_stride"] != 1:
                c["ksize"] = 1
                c["stride"] = c["block_stride"]
                c["conv_filters_out"] = filters_out
                shortcut = conv(shortcut, c)
                shortcut = bn(shortcut, c)

            return activation(x + shortcut)

    def conv(x, c):
        ksize = c["ksize"]
        stride = c["stride"]
        filters_out = c["conv_filters_out"]
        filters_in = x.get_shape()[-1] #Depth
        shape = [ksize, ksize, filters_int, filters_out]
        initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
        weights = _get_variable("weights", shape=shape, dtype="float", initializer=initializer, weight_decay=CONV_WEIGHT_DECAY)

    def bn(x, c):
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]
        if c["use_bias"]:
            bias = _get_variable("bias", params_shape, initializer=tf.zeros_initializer)
            return x + bias
        axis = list(range(len(x_shape) - 1))
        beta = _get_variable("beta", params_shape, initializer=tf.zeros_initializer)
        gamma = _get_variable("gamma", params_shape, initializer=tf.zeros_initializer)
        moving_mean = get_variable("moving_mean", params_shape, initializer=tf.zeros_initializer, trainable=False)
        moving_variance = _get_variable("moving_variance", params_shape, initializer=tf.zeros_initializer, trainable=False)
        mean, variance = tf.nn.monents(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
        mean, variance = control_flow_ops.cond(c["is_traning"], lambda: (mean, variance), lambda: (moving_mean, moving_variance))
        x =tf.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
        return x 

    def _max_pool(x, ksize=3, stride=2):
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding="SAME")

    def fc(x, c):
        num_units_in = x.get_shape()[-1]
        num_units_out = c["fc_units_out"]
        weights_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)
        weights = _get_varianbe("weights", shape=[num_units_in, num_units_out], initializer=weights_initializer, weight_decay=FC_WEIGHT_DECAY)
        x = tf.nn.xw_plus_b(x, weights, biases)
        return x

    def _get_variable(name, shape, initializer, weight_decay=0.0, dtype="float", trainable=True):
        if weight_decay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            retularizer = None
        collections = [tf.GraphKeys.VARIABLES, RESNET_VARIABLES]
        return tf.get_variable(name, shape, initializer=initializer, dtype=dtype, regularizer=regularizer, collections=collections, trainable=trainable)

    def loss(logits, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        regularization_losses = tf.get_collections(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
        tf.scalar_summary("loss", loss_)
        return loss_
class Inference():
    def __init__(self, RawTensor):
        print RawTensor
        Depth_major = tf.reshape(RawTensor, [-1, DEPTH, HEIGHT, WIDTH])
        print Depth_major
        InputTensor = tf.transpose(Depth_major, [0, 2, 3, 1])
        print InputTensor
        self.model = Models()
        self.inference = self.model.ResNet(InputTensor)
        

    def loss(self, logits, labels):
        #labels = tf.cast(labels, tf.int64)
        print 'logits', logits
        print 'labels', labels
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(
            cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def add_loss_summaries(self, total_loss):
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        for l in losses + [total_loss]:
            tf.summary.scalar(l.op.name + ' (raw) ', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))
        return loss_averages_op

    def Train(self, total_loss, global_step, init_lr=INITIAL_LEARNING_RATE, optimizer='Adam'):
        #num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        #decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        decay_steps = LEARNING_RATE_DECAY_STEP
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE , global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)
        #lr = tf.constant(init_lr)
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

    def Accuracy(self, logits, labels):
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)
        return accuracy

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

