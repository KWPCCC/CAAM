 # coding=utf-8
"""Implementation of CAAM attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
# import cv2
import pandas as pd
import scipy.stats as st
from scipy.misc import imread, imsave
from tensorflow.contrib.image import transform as images_transform
from tensorflow.contrib.image import rotate as images_rotate

import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

import random

slim = tf.contrib.slim

tf.flags.DEFINE_integer('batch_size', 10, 'How many images process at one time.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'max epsilon.')

tf.flags.DEFINE_integer('num_iter', 10, 'max iteration.')

tf.flags.DEFINE_float('momentum', 1.0, 'momentum about the model.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_float('prob', 0.5, 'probability of using diverse inputs.')

tf.flags.DEFINE_integer('image_resize', 331, 'Height of each input images.')

tf.flags.DEFINE_string('checkpoint_path', './models',
                       'Path to checkpoint for pretained models.')

tf.flags.DEFINE_string('input_dir', './dev_data/val_rs',
                       'Input directory with images.')

tf.flags.DEFINE_string('output_dir', './only-ct3',
                       'Output directory with images.')

FLAGS = tf.flags.FLAGS

np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)

model_checkpoint_map = {
    'inception_v3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'adv_inception_v3_rename.ckpt'),
    'ens3_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
    'ens4_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
    'inception_v4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
    'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'ens_adv_inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    'resnet_v2': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_101.ckpt')}


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


kernel = gkern(7, 3).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)

def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    # image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    # out = np.uint8(out * 255)
    # cv.imshow("gasuss", out)
    return out

def shuffled(images):
    x_d1, x_d2, x_d3, x_d4, x_d5 = np.zeros(shape=images.shape, dtype='float32'),np.zeros(shape=images.shape, dtype='float32'),np.zeros(shape=images.shape, dtype='float32'),np.zeros(shape=images.shape, dtype='float32'),np.zeros(shape=images.shape, dtype='float32')
    for i in range(len(images)):
        img = images[i]
        R = img[::, ::, 0]
        G = img[::, ::, 1]
        B = img[::, ::, 2]
        x_d1[i] = np.stack([R, B, G], axis=-1)
        x_d2[i] = np.stack([B, G, R], axis=-1)
        x_d3[i] = np.stack([B, R, G], axis=-1)
        x_d4[i] = np.stack([G, R, B], axis=-1)
        x_d5[i] = np.stack([G, B, R], axis=-1)

    return x_d1, x_d2, x_d3, x_d4, x_d5

def channel_transformation(x_test):
    newxlist = []
    for n in range(3):
        padding = np.random.uniform(-1, 1, size=(299, 299))
        np.random.shuffle(padding)
        newx = np.zeros(shape=(len(x_test), 299, 299, 3))
        for i in range(len(x_test)):
            R = x_test[i][::, ::, 0]
            G = x_test[i][::, ::, 1]
            B = x_test[i][::, ::, 2]
            set = [padding,R,G,B]
            rnew = random.sample(set, 1)
            gnew = random.sample(set, 1)
            bnew = random.sample(set, 1)
            newx[i] = np.stack([rnew, gnew, bnew], axis=-1)
        newxlist.append(newx)
    return newxlist

def channel_patch(x_test):
    newxlist = []
    for n in range(5):
        newx = np.zeros(shape=(len(x_test), 299, 299, 3))
        for i in range(len(x_test)):
            R = x_test[i][::, ::, 0]
            G = x_test[i][::, ::, 1]
            B = x_test[i][::, ::, 2]
            set = [1/2,1/4,1/8,1/16]
            rnew = random.sample(set, 1)
            gnew = random.sample(set, 1)
            bnew = random.sample(set, 1)
            h, w = x_test[i].shape[:2]
            x = random.randint(0, w // 2)
            y = random.randint(0, h // 2)
            rect_w = random.randint(w // 4, w // 2)
            rect_h = random.randint(h // 4, h // 2)
            rect = (x, y, rect_w, rect_h)
            R[y:y + rect_h, x:x + rect_w] = R[y:y + rect_h, x:x + rect_w]*rnew
            G[y:y + rect_h, x:x + rect_w] = G[y:y + rect_h, x:x + rect_w] * gnew
            B[y:y + rect_h, x:x + rect_w] = B[y:y + rect_h, x:x + rect_w] * bnew
            newx[i] = np.stack([R, G, B], axis=-1)
        newxlist.append(newx)
    return newxlist

def get_expand_imgs(x_test):
    # padding = np.zeros(shape=(299, 299), dtype='float32')
    padding = np.random.uniform(-1, 1, size=(299, 299))
    np.random.shuffle(padding)
    x_test_R = np.zeros(shape=(len(x_test), 299, 299, 3))
    x_test_G = np.zeros(shape=(len(x_test), 299, 299, 3))
    x_test_B = np.zeros(shape=(len(x_test), 299, 299, 3))
    x_test_RG = np.zeros(shape=(len(x_test), 299, 299, 3))
    x_test_GB = np.zeros(shape=(len(x_test), 299, 299, 3))
    x_test_RB = np.zeros(shape=(len(x_test), 299, 299, 3))
    for i in range(len(x_test)):
        R = x_test[i][::, ::, 0]
        G = x_test[i][::, ::, 1]
        B = x_test[i][::, ::, 2]
        x_test_R[i] = np.stack([R, padding, padding], axis=-1)
        x_test_G[i] = np.stack([padding, G, padding], axis=-1)
        x_test_B[i] = np.stack([padding, padding, B], axis=-1)
        x_test_RG[i] = np.stack([R, G, padding], axis=-1)
        x_test_GB[i] = np.stack([padding, G, B], axis=-1)
        x_test_RB[i] = np.stack([R, padding, B], axis=-1)
    return x_test, x_test_R, x_test_G, x_test_B,x_test_RG, x_test_GB, x_test_RB

def get_expand_imgs_new(x_test):
    # padding = np.zeros(shape=(299, 299), dtype='float32')
    padding = np.random.uniform(-1, 1, size=(299, 299))
    np.random.shuffle(padding)
    x_test_RG = np.zeros(shape=(len(x_test), 299, 299, 3))
    x_test_GB = np.zeros(shape=(len(x_test), 299, 299, 3))
    x_test_RB = np.zeros(shape=(len(x_test), 299, 299, 3))
    for i in range(len(x_test)):
        R = x_test[i][::, ::, 0]
        G = x_test[i][::, ::, 1]
        B = x_test[i][::, ::, 2]
        x_test_RG[i] = np.stack([R, G, padding], axis=-1)
        x_test_GB[i] = np.stack([padding, G, B], axis=-1)
        x_test_RB[i] = np.stack([R, padding, B], axis=-1)
    return x_test, x_test_RG, x_test_GB, x_test_RB

def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def randx(x, alpha):
    return x + alpha*tf.random_normal(x.shape, mean=0, stddev=1), x - alpha*tf.random_normal(x.shape, mean=0, stddev=1)




def image_augmentation(x):
    # [32, 299, 299, 3]
    # img, noise
    # batch_size行1列
    # 32,1
    one = tf.fill([tf.shape(x)[0], 1], 1.)
    zero = tf.fill([tf.shape(x)[0], 1], 0.)
    # 32,8
    # 1 , 0, 0, 0 ,1 ,0 ,0, 0
    transforms = tf.concat([one, zero, zero, zero, one, zero, zero, zero], axis=1)
    # 32,6 -> 32,8
    rands = tf.concat([tf.truncated_normal([tf.shape(x)[0], 6], stddev=0.05), zero, zero], axis=1)
    # 32, 32, 8
    return images_transform(x, transforms + rands, interpolation='BILINEAR')


def image_rotation(x):
    """ imgs, scale, scale is in radians """
    rands = tf.truncated_normal([tf.shape(x)[0]], stddev=0.05)
    return images_rotate(x, rands, interpolation='BILINEAR')

def xueruo(images):
    x_d1, x_d2, x_d3, x_d4, x_d5, x_d6 = np.zeros(shape=images.shape, dtype='float32'),np.zeros(shape=images.shape, dtype='float32'),np.zeros(shape=images.shape, dtype='float32'),np.zeros(shape=images.shape, dtype='float32'),np.zeros(shape=images.shape, dtype='float32'),np.zeros(shape=images.shape, dtype='float32')
    for i in range(len(images)):
        img = images[i]
        R = img[::, ::, 0]
        G = img[::, ::, 1]
        B = img[::, ::, 2]
        x_d1[i] = np.stack([R, G/2, B/4], axis=-1)
        x_d2[i] = np.stack([R, G/4, B/2], axis=-1)
        x_d3[i] = np.stack([R/2, G, B/4], axis=-1)
        x_d4[i] = np.stack([R/4, G, B/2], axis=-1)
        x_d5[i] = np.stack([R/2, G/4, B], axis=-1)
        x_d6[i] = np.stack([R/4, G/2, B], axis=-1)

    return x_d1, x_d2, x_d3, x_d4, x_d5, x_d6


def input_diversity(input_tensor):
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_height, FLAGS.image_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ret


def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    f2l = load_labels('./dev_data/val_rs.csv')
    eps = 2 * FLAGS.max_epsilon / 255.0
    num_classes = 1001
    momentum = FLAGS.momentum
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    check_or_create_dir(FLAGS.output_dir)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_image = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_image + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_image - eps, -1.0, 1.0)
        x_batch = tf.concat([x_input, x_input / 2, x_input / 4, x_input / 8, x_input / 16], axis=0)
        y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        accumulated_grad_ph = tf.placeholder(dtype=tf.float32, shape=batch_shape)
        noiseall_ph = tf.placeholder(dtype=tf.float32, shape=batch_shape)
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_v3, end_points_v3 = inception_v3.inception_v3(
                x_input, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_v31, end_points_v31 = inception_v3.inception_v3(
                x_batch, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

        pred = tf.argmax(end_points_v3['Predictions'], 1)
        predy = pred[:y.shape[0]]

        one_hot = tf.concat([tf.one_hot(y, num_classes)] * 5, axis=0)

        cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits_v31)
        noise = tf.reduce_sum(
            tf.split(tf.gradients(cross_entropy, x_batch)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:,
                                                                   None,
                                                                   None, None, None], axis=0)

        # noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
        noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)


        noisem = momentum * accumulated_grad_ph + noiseall_ph
        adv_input_update = x_input + alpha * tf.sign(noisem)
        adv_input_update = tf.clip_by_value(adv_input_update, x_min, x_max)

        set_input = tf.placeholder(tf.float32, shape=batch_shape)
        setgrad = tf.placeholder(dtype=tf.float32, shape=batch_shape)
        set_input_update = set_input + alpha * tf.sign(setgrad)
        set_input_update = tf.clip_by_value(set_input_update, x_min, x_max)

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        # s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        # s3 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        # s4 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))
        # s5 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        # s6 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        # s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        # s8 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))



        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            s1.restore(sess, model_checkpoint_map['inception_v3'])
            # s2.restore(sess, model_checkpoint_map['inception_v4'])
            # s3.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            # s4.restore(sess, model_checkpoint_map['resnet_v2'])
            # s5.restore(sess, model_checkpoint_map['ens3_adv_inception_v3'])
            # s6.restore(sess, model_checkpoint_map['ens4_adv_inception_v3'])
            # s7.restore(sess, model_checkpoint_map['ens_adv_inception_resnet_v2'])
            # s8.restore(sess, model_checkpoint_map['adv_inception_v3'])

            idx = 0
            l2_diff = 0
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                idx = idx + 1
                print("start the i={} attack".format(idx))

                # data_hlist = channel_patch(np.copy(images))
                data_hlist = channel_transformation(np.copy(images))
                data_hlist.append(images)
                # data_hlist = []
                # # data_hlist.append(images)
                # data_hlist.append(img1)
                # data_hlist.append(img2)
                # data_hlist.append(img3)
                # data_hlist.append(img4)
                # data_hlist.append(img5)
                # data_hlist.append(img6)
                y1 = sess.run([predy], feed_dict={x_input: images})
                noise2 = np.zeros(shape=batch_shape)
                adv = images
                for i in range(num_iter):
                    noiseall = np.zeros(shape=batch_shape)
                    for l in data_hlist:
                        noise1 = sess.run([noise], feed_dict={x_input: l, y: y1[0]})
                        noiseall = noiseall + np.squeeze(noise1)

                    adv, noise2 = sess.run([adv_input_update, noisem], feed_dict={x_input:adv,x_image:images,  accumulated_grad_ph: noise2, noiseall_ph: noiseall})
                    adv = np.clip(adv, images - eps, images + eps)

                    newdata_hlist = []
                    for l in range(len(data_hlist)):
                        # data_hlistl = sess.run([set_input_update], feed_dict={set_input: data_hlist[l], setgrad: noise2})
                        # newdata_hlist.append(data_hlistl)
                        data_hlistl = np.clip(data_hlist[l]+alpha*np.sign(noise2), images - eps, images + eps)
                        newdata_hlist.append(data_hlistl)
                    data_hlist = newdata_hlist
                save_images(adv, filenames, FLAGS.output_dir)
                diff = (adv + 1) / 2 * 255 - (images + 1) / 2 * 255
                l2_diff += np.mean(np.linalg.norm(np.reshape(diff, [-1, 3]), axis=1))

            print('{:.2f}'.format(l2_diff * FLAGS.batch_size / 1000))#19.46


def load_labels(file_name):
    import pandas as pd
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    return f2l










if __name__ == '__main__':
    tf.app.run()
