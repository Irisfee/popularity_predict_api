import argparse
from math import floor, log2
from os import fsync, mkdir, path
from shutil import rmtree
from sys import stdout
import numpy as np
import tensorflow as tf
from scipy.stats import kendalltau, spearmanr
from tensorflow.contrib import learn
MEL_BIN = 128
FRAME_NUM = 323
def leaky_relu(x, alpha=0.1):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


NON_LINEAR = leaky_relu

def make_tag_batch(tg, trix, pos, b_size):
    ids = trix[pos:pos + b_size]
    b_tg = np.take(tg, ids, axis=0)
    return b_tg

def make_batch(feat, pt, trix, pos, b_size):
    ids = trix[pos:pos + b_size]
    b_f = np.take(feat, ids, axis=0)
    b_pt = np.take(pt, ids, axis=0) if pt is not None else None

    return b_f, b_pt
    

def conv1_128_4_gen(x):
    return tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[128, 4],
        activation=NON_LINEAR)


def cnn(x, dr, mode, stt):
    # 128*323*1

    # 1*320*32
    return cnn_body(conv1_128_4_gen(x), dr, mode, stt)


def inception_cnn(x, dr, mode, stt):
    # 128*323*1

    # 1*320*32
    conv1_128_4 = conv1_128_4_gen(x)

    # 132*327*1
    pad1_132_8 = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]])
    # 1*320*16
    conv1_132_8 = tf.layers.conv2d(
        inputs=pad1_132_8,
        filters=16,
        kernel_size=[132, 8],
        activation=NON_LINEAR)

    # 140*335*1
    pad1_140_16 = tf.pad(x, [[0, 0], [6, 6], [6, 6], [0, 0]])
    # 1*320*16
    conv1_140_16 = tf.layers.conv2d(
        inputs=pad1_140_16,
        filters=16,
        kernel_size=[140, 16],
        activation=NON_LINEAR)

    # 1*320*64
    concat = tf.concat([conv1_128_4, conv1_132_8, conv1_140_16], axis=3)
    # 1*320*32
    conv1 = tf.layers.conv2d(
        inputs=concat,
        filters=32,
        kernel_size=[1, 1],
        activation=NON_LINEAR)

    return cnn_body(conv1, dr, mode, stt)


def cnn_body(head, dr, mode, stt):
    # 1*320*32

    logits_all = {}

    # 1*160*32
    pool1 = tf.layers.max_pooling2d(
        inputs=head, pool_size=[1, 2], strides=[1, 2])

    # 1*157*64
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[1, 4],
        activation=NON_LINEAR)

    # 1*78*64
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=[1, 2], strides=[1, 2])
    drop1 = tf.layers.dropout(
        inputs=pool2, rate=dr, training=(mode == learn.ModeKeys.TRAIN))

    # 1*78*256
    conv3 = tf.layers.conv2d(
        inputs=drop1,
        filters=256,
        kernel_size=[1, 1],
        activation=NON_LINEAR)
    drop2 = tf.layers.dropout(
        inputs=conv3, rate=dr, training=(mode == learn.ModeKeys.TRAIN))
    conv4 = tf.layers.conv2d(
        inputs=drop2,
        filters=256,
        kernel_size=[1, 1],
        activation=NON_LINEAR)
    drop3 = tf.layers.dropout(
        inputs=conv4, rate=dr, training=(mode == learn.ModeKeys.TRAIN))

    # 1*78*1
    conv5 = tf.layers.conv2d(
        inputs=drop3,
        filters=1,
        kernel_size=[1, 1],
        activation=NON_LINEAR)

    # 1
    logits_pt = tf.reduce_mean(
        conv5, axis=[1, 2], name='GlobalAveragePooling')
    #1 and 2 dimension is reduced to mean and return 1d array,length of array is batchsize

    logits_all['pt'] = logits_pt
    logits_all['emb'] = conv5
    return logits_all


def tag_regression_clsf(tags, dr, mode):
    dense1 = tf.layers.dropout(
        inputs=tf.layers.dense(
            inputs=tags,
            units=100,
            activation=NON_LINEAR,
            name='dense1'),
        rate=dr,
        training=(mode == learn.ModeKeys.TRAIN))
    dense2 = tf.layers.dropout(
        inputs=tf.layers.dense(
            inputs=dense1,
            units=100,
            activation=NON_LINEAR,
            name='dense2'),
        rate=dr,
        training=(mode == learn.ModeKeys.TRAIN))
    dense3 = tf.layers.dropout(
        inputs=tf.layers.dense(
            inputs=dense2,
            units=30,
            activation=NON_LINEAR,
            name='dense3'),
        rate=dr,
        training=(mode == learn.ModeKeys.TRAIN))
    logits_tag = tf.layers.dense(
        inputs=dense3,
        units=1,
        activation=NON_LINEAR,
        name='logits_tag')

    return logits_tag

def score_pred_only(args):
    print ('Pred only.')
    stdout.flush()

    train_set = args.train_set
    batch_size = int(args.batch_size)
    model_type = args.model_type
    stt = args.secondary_target_type
    use_tag = args.use_tag
    tag_type = args.tag_type
    dr_rate = float(args.dropout_rate)
    model_dir = '{}.mdl/'.format(args.output)
    loss_t_w = float(args.tagging_loss_weight)
    pred_f = 'ret30_pred.npy'
    pred_emb = 'ret30_pred.emb.npy'

    test_feat = np.load(
        path.join(train_set, 'xte.npy')).reshape(-1, MEL_BIN, FRAME_NUM, 1)
    y_num = test_feat.shape[0]

    test_pt, y_size, emb_size = None, 1, 78
    test_tags = None
    if use_tag:
        test_tags = np.load(
            path.join(train_set, 'test_tgte.{}.npy'.format(tag_type)))

    x_f = tf.placeholder(tf.float32, [None, MEL_BIN, FRAME_NUM, 1])
    y_t = tf.placeholder(tf.float32, [None, y_size])
    emb_t = tf.placeholder(tf.float32, [None, emb_size, 1])
    mode = tf.placeholder(tf.string)  # TRAIN, EVAL, INFER
    tags = None
    if use_tag:
        tags = tf.placeholder(tf.float32, [None, TAG_SIZE])

    logits_all = None
    if model_type == 'incept':
        print ('Model type: Inception CNN.\n')
        logits_all = inception_cnn(
            x_f, dr_rate, mode, stt)
    else:
        print ('Model type: Plain CNN.\n')
        logits_all = cnn(
            x_f, dr_rate, mode, stt)

    logits_pt = logits_all['pt']  # primary target(s)
    logits_emb = logits_all['emb']
    if use_tag:
        logits_tag = tag_regression_clsf(
            tags, dr_rate, mode)
        logits_pt += loss_t_w * logits_tag

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, path.join(model_dir, 'model.ckpt'))
        test_logitss, test_embs = None, None
        for test_pos in range(0, test_feat.shape[0], batch_size):
            b_f, b_y = make_batch(
                test_feat, test_pt, np.arange(y_num), test_pos, batch_size)
            if use_tag:
                b_tg = make_tag_batch(
                    test_tags, np.arange(y_num), test_pos, batch_size)
                test_logits, test_emb = sess.run(
                    [logits_pt, logits_emb],
                    feed_dict={
                        x_f: b_f, tags: b_tg,
                        mode: learn.ModeKeys.INFER})
            else:
                test_logits, test_emb = sess.run(
                    [logits_pt, logits_emb],
                    feed_dict={
                        x_f: b_f,
                        mode: learn.ModeKeys.INFER})
            test_emb = test_emb.reshape(-1, emb_size)
            if test_logitss is None:
                test_logitss = test_logits  # * NORM_FACTOR
            else:
                test_logitss = np.concatenate(
                    (test_logitss, test_logits), axis=0)
            if test_embs is None:
                test_embs = test_emb  # * NORM_FACTOR
            else:
                test_embs = np.concatenate(
                    (test_embs, test_emb), axis=0)
    print (test_logitss)
