from __future__ import print_function
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange
import os
import scipy.misc as misc
import PIL.Image

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "10", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "test", "Mode train/ test/ visualize")


test_file_path = 'images/'

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e3 + 1)
# MAX_ITERATION = 2000
NUM_OF_CLASSESS = 2
IMAGE_SIZE = 500


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))
    loss_summary = tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()

    # create two summary writers to show training loss and validation loss in the same graph
    # need to create two folders 'train' and 'validation' inside FLAGS.logs_dir
    train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/validation')

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, loss_summary], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                train_writer.add_summary(summary_str, itr)

            if itr % 10 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss, summary_sva = sess.run([loss, loss_summary], feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                # add validation loss to TensorBoard
                validation_writer.add_summary(summary_sva, itr)
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    # elif FLAGS.mode == "visualize":
    #     valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
    #     pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
    #                                                 keep_probability: 1.0})
    #     valid_annotations = np.squeeze(valid_annotations, axis=3)
    #     pred = np.squeeze(pred, axis=3)
    #
    #     for itr in range(FLAGS.batch_size):
    #         utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
    #         utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
    #         utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
    #         print("Saved image: %d" % itr)

    elif FLAGS.mode == "visualize":
        # num: the number of images to be tested which can be a single batch_size or all validation set
        valid_images, valid_annotations, num = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        print(valid_images)
        print(type(valid_images))
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, keep_probability: 1.0})
        pred = np.squeeze(pred, axis=3)
        print(len(valid_images))
        print(num)


        # for f in file_list:
        #     filename = test_file_path + '/' + f
        #     print(filename)
        #     image = misc.imread(filename)
        #     print(len(image.shape))
        #     if len(image.shape) < 3:  # make sure images are of shape(h,w,3)
        #         image = np.array([image for i in range(3)])
        #     print(image)
        #     image_list.append(image)
        # image_list = np.array(image_list)


        for itr in range(num):
            src_img = valid_images[itr].astype(np.uint8)
            pred_img = pred[itr].astype(np.uint8)
            print(pred_img)
            # create fliedir

            dirs = 'test_visualize/'

            if not os.path.exists(dirs):
                os.makedirs(dirs)

            # create file

            file_object = open('test_visualize/' + "inp_" + str(itr) + '.png', 'w')
            file_object.close()
            file_object = open('test_visualize/' + "pred_" + str(itr) + '.png', 'w')
            file_object.close()

            # save images to ./logs/test_visualize
            utils.save_image(src_img, 'test_visualize/', name="inp_" + str(itr) + '.png')

            utils.save_image(pred_img, 'test_visualize/', name="pred_" + str(itr) + '.png')
            for i in range(pred_img.shape[0]):
                for j in range(pred_img.shape[1]):
                    if pred_img[i, j] != 0:
                        # if your source images are RGB format, you need to change three channels
                        src_img[i, j] = 200
            utils.save_image(src_img, 'test_visualize/', name="visual_" + str(itr) + '.png')

            # 标签可视化
            for i in range(pred_img.shape[0]):
                for j in range(pred_img.shape[1]):
                    if pred_img[i, j] != 0:
                        pred_img[i, j] = 200

            file_object = open('test_visualize/' + "pred_visual" + str(itr) + '.png', 'w')
            file_object.close()
            utils.save_image(pred_img, 'test_visualize/', name="pred_visual" + str(itr) + '.png')


            print("Saved image: %d" % itr)

    elif FLAGS.mode == 'test':

        file_list = []
        rootdir = test_file_path
        for parent, dirnames, filenames in os.walk(rootdir):
            for filename in filenames:
                print("Parent folder:", parent)
                print("Filename:", filename)
                file_list.append(os.path.join(parent, filename))
        image_list = np.array([misc.imresize(misc.imread(f),
                                         [IMAGE_SIZE, IMAGE_SIZE], interp='nearest') for f in file_list])

        num = len(file_list)
        print(num)

        # for f in file_list:
        #     filename = f
        #     print(filename)
        #     image = misc.imread(filename)
        #     print(len(image.shape))
        #     if len(image.shape) < 3:  # make sure images are of shape(h,w,3)
        #         image = np.array([image for i in range(3)])
        #     print(image)
        #     image_list = np.append(image_list, image)
        # image_list = np.array(image_list)

        indexes = range(image_list.shape[0])
        image_list_final = image_list[indexes]
        print(image_list_final)
        print(type(image_list_final))
        pred = sess.run(pred_annotation, feed_dict={image: image_list_final, keep_probability: 1.0})
        pred = np.squeeze(pred, axis=3)

        print(num)
        for itr in range(num):
            src_img = image_list[itr].astype(np.uint8)
            pred_img = pred[itr].astype(np.uint8)
            print(pred_img)


            # create fliedir

            dirs = 'test/'

            if not os.path.exists(dirs):
                os.makedirs(dirs)

            # create file

            file_object = open('test/' + "inp_" + str(itr) + '.png', 'w')
            file_object.close()
            file_object = open('test/' + "pred_" + str(itr) + '.png', 'w')
            file_object.close()
            # save images to ./logs/test_visualize
            utils.save_image(src_img, 'test/', name="inp_" + str(itr) + '.png')
            utils.save_image(pred_img, 'test/', name="pred_" + str(itr) + '.png')
            for i in range(pred_img.shape[0]):
                for j in range(pred_img.shape[1]):
                    if pred_img[i, j] != 0:
                        # if your source images are RGB format, you need to change three channels
                        src_img[i, j] = 200
            utils.save_image(src_img, 'test/', name="visual_" + str(itr) + '.png')

            # 标签可视化
            for i in range(pred_img.shape[0]):
                for j in range(pred_img.shape[1]):
                    if pred_img[i, j] != 0:
                        pred_img[i, j] = 200

            file_object = open('test/' + "pred_visual" + str(itr) + '.png', 'w')
            file_object.close()
            utils.save_image(pred_img, 'test/', name="pred_visual" + str(itr) + '.png')


            print("Saved image: %d" % itr)



if __name__ == "__main__":
    tf.app.run()
