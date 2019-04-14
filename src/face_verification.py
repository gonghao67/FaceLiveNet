from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import math
from datetime import datetime
from PIL import Image

sys.path.append('../')
import align.face_align_mtcnn
import facenet
from scipy import spatial
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import importlib
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

import time
from datetime import datetime
import lfw

def exp_forward(args):
    network = importlib.import_module(args.model_def, 'inference')

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    random.seed(args.seed)

    train_set = facenet.get_dataset(args.data_dir)
    image_list, label_list, usage_list, nrof_classes = facenet.get_image_paths_and_labels_fer2013(args.data_dir,
                                                                                                  args.labels_expression,
                                                                                                  'Training')

    
    print('Total number of subjects: %d' % nrof_classes)
    print('Total number of images: %d' % len(image_list))
    

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.pretrained_model))
        print('Pre-trained model: %s' % pretrained_model)

    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)

    if args.evaluate_express:
        print('FER2013 test data directory: %s' % args.data_dir)
        fer2013_paths_test, label_list_test, usage_list_test, nrof_classes_test = facenet.get_image_paths_and_labels_fer2013(
            args.data_dir,
            args.labels_expression,
            'PublicTest')

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)

        # Create a queue that produces indices into the image_list and label_list
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                                                    shuffle=True, seed=None, capacity=32)

        index_dequeue_op = index_queue.dequeue_many(args.batch_size * args.epoch_size, 'index_dequeue')

        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')

        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 1), name='labels')

        keep_probability_placeholder = tf.placeholder(tf.float32, name='keep_probability')

        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(1,), (1,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder], name='enqueue_op')

        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unpack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_png(file_contents)
                if args.random_rotate:
                    image = tf.py_func(facenet.random_rotate_image, [image], tf.uint8)
                if args.random_crop:
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)

                # pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])

        image_batch, label_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder,
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        # image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')

        print('Building training graph')

        # Build the inference graph
        prelogits, _ = network.inference(image_batch, keep_probability_placeholder,
                                         phase_train=phase_train_placeholder, weight_decay=args.weight_decay)
        # logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None,
        #         weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        #         weights_regularizer=slim.l2_regularizer(args.weight_decay),
        #         scope='Logits', reuse=False)

        logits0 = slim.fully_connected(prelogits, 512, activation_fn=tf.nn.relu,
                                       weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                       weights_regularizer=slim.l2_regularizer(args.weight_decay),
                                       scope='Logits0', reuse=False)

        logits = slim.fully_connected(logits0, len(set(label_list)), activation_fn=tf.nn.relu,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(args.weight_decay),
                                      scope='Logits', reuse=False)

        logits = tf.identity(logits, 'logits')

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Add center loss
        if args.center_loss_factor > 0.0:
            # prelogits_center_loss, centers, _, centers_cts_batch_reshape = facenet.center_loss(prelogits, label_batch, args.center_loss_alfa, nrof_classes)
            prelogits_center_loss, centers, _, centers_cts_batch_reshape = facenet.center_loss(embeddings, label_batch,
                                                                                               args.center_loss_alfa,
                                                                                               nrof_classes)
            # prelogits_center_loss, _ = facenet.center_loss_similarity(prelogits, label_batch, args.center_loss_alfa, nrof_classes) ####Similarity cosine distance, center loss
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   args.learning_rate_decay_epochs * args.epoch_size,
                                                   args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, label_batch, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')
        total_loss = tf.add_n([cross_entropy_mean], name='total_loss')

        #### Training accuracy of softmax: check the underfitting or overfiting #############################
        correct_prediction = tf.equal(tf.argmax(logits, 1), label_batch)
        softmax_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        ########################################################################################################

        ########## edit mzh   #####################
        # Create list with variables to restore
        restore_vars = []
        update_gradient_vars = []
        # logits_var = []
        if args.pretrained_model:
            for var in tf.global_variables():
                #     if 'InceptionResnet' in var.op.name:
                #         restore_vars.append(var)
                if 'Logits' in var.op.name or 'Logits0' in var.op.name:
                    print(var.op.name)
                    update_gradient_vars.append(var)
            restore_vars = tf.global_variables();
            restore_saver = tf.train.Saver(restore_vars)
        else:
            update_gradient_vars = tf.trainable_variables()

        # update_gradient_vars0 = tf.global_variables()
        # update_gradient_vars = tf.trainable_variables()
        train_op = facenet.train(total_loss, global_step, args.optimizer,
                                 learning_rate, args.moving_average_decay, update_gradient_vars, args.log_histograms)

        # saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        tf.train.start_queue_runners(sess=sess)

        with sess.as_default():

            if pretrained_model:
                print(
                'Restoring pretrained model: %s' % os.path.join(os.path.expanduser(args.pretrained_model), ckpt_file))
                # saver.restore(sess, pretrained_model)
                restore_saver.restore(sess, os.path.join(os.path.expanduser(args.pretrained_model), ckpt_file))

            # Training and validation loop
            print('Running training')
            epoch = 0
            acc = 0
            val = 0
            far = 0
            best_acc = 0
            acc_expression = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                print('Epoch step: %d' % step)
                epoch = step // args.epoch_size
                
                if not (epoch % 1):
                    if args.evaluate_express:
                        acc_expression = evaluate_expression(sess, enqueue_op, image_paths_placeholder,
                                                             labels_placeholder,
                                                             phase_train_placeholder, batch_size_placeholder, logits,
                                                             label_batch, fer2013_paths_test, label_list_test,
                                                             args.lfw_batch_size,
                                                             log_dir, step, summary_writer,
                                                             keep_probability_placeholder)
               
    return model_dir

def evaluate_expression(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
             batch_size_placeholder,
             logits, labels, image_paths, actual_expre, batch_size, log_dir, step, summary_writer,
             keep_probability_placeholder):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on FER2013 images')
    nrof_images = len(actual_expre)
    nrof_batches = nrof_images // batch_size

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.arange((nrof_batches*batch_size)), 1) ## used for noting the locations of the image in the queue
    image_paths_array = np.expand_dims(np.array(image_paths[0:nrof_batches*batch_size]), 1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})


    logits_size = logits.get_shape()[1]


    logits_array = np.zeros((nrof_batches*batch_size, logits_size), dtype=float)
    lab_array = np.zeros((nrof_batches*batch_size), dtype=int)
    for ii in range(nrof_batches):
        #print('nrof_batches %d'%ii)
        feed_dict = {phase_train_placeholder: False, batch_size_placeholder: batch_size,
                     keep_probability_placeholder: 1.0}
        logits_batch, lab = sess.run([logits, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        logits_array[lab] = logits_batch
    assert np.array_equal(lab_array, np.arange(nrof_batches*batch_size)) == True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'

    express_probs = np.exp(logits_array) / np.tile(np.reshape(np.sum(np.exp(logits_array), 1), (logits_array.shape[0], 1)), (1, logits_array.shape[1]))
    nrof_expression = express_probs.shape[1]
    expressions_predict = np.argmax(express_probs, 1)
    #### Training accuracy of softmax: check the underfitting or overfiting #############################
    correct_prediction = np.equal(expressions_predict, actual_expre[0:nrof_batches*batch_size])
    test_expr_acc = np.mean(correct_prediction)

    ########################################################################################################
    print('%d expressions recognition accuracy is: %f' % (nrof_expression, test_expr_acc))

    fer_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='fer/accuracy', simple_value=test_expr_acc)
    summary.value.add(tag='time/fer', simple_value=fer_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir, 'Fer2013_result.txt'), 'at') as f:
        f.write('%d\t%.5f\n' % (step, test_expr_acc))

    return test_expr_acc


def load_align_model(args):
    with tf.device('/cpu:0'):

        pnet, rnet, onet = align.face_align_mtcnn.load_align_mtcnn(args.align_model_dir)

    return pnet, rnet, onet

def load_face_verif_model(args):
    with tf.device('/cpu:0'):

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Load the model of face verification
        print('Model directory: %s' % args.model_dir)
        meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        model_dir_exp = os.path.expanduser(args.model_dir)
        saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
        #saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))
        saver.restore(sess, os.path.join(model_dir_exp, ckpt_file))

    return sess


def load_models(args):
    with tf.device('/cpu:0'):
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)

        # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count = {'CPU': 0})) as sess:
        # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True)) as sess:
        #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        # Load the model of face detection
        pnet, rnet, onet = align.face_align_mtcnn.load_align_mtcnn(args.align_model_dir)

        # Load the model of face verification
        print('Model directory: %s' % args.model_dir)
        meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        model_dir_exp = os.path.expanduser(args.model_dir)
        saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
        #saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))
        saver.restore(sess, os.path.join(model_dir_exp, ckpt_file))

    return pnet, rnet, onet, sess

def load_models_forward(args, nrof_expressions):
    with tf.device('/cpu:0'):
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)

        # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count = {'CPU': 0})) as sess:
        # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True)) as sess:
        # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:


        # Load the model of face detection
        pnet, rnet, onet = align.face_align_mtcnn.load_align_mtcnn(args.align_model_dir)


        ########### load face verif_expression model #############################
        # Load the model of face verification
        print('Model directory: %s' % args.model_dir)
        meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))

        network = importlib.import_module(args.model_def, 'inference')

        image_batch_placeholder = tf.placeholder(tf.float32, shape=(None, args.image_size, args.image_size, 3), name='image_batch')

        # Build the inference graph
        prelogits, _ = network.inference(image_batch_placeholder, 1.0,
                                         phase_train=False, weight_decay=args.weight_decay)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')


        logits0 = slim.fully_connected(prelogits, 512, activation_fn=tf.nn.relu,
                                       weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                       weights_regularizer=slim.l2_regularizer(args.weight_decay),
                                       scope='Logits0', reuse=False)

        logits = slim.fully_connected(logits0, nrof_expressions, activation_fn=tf.nn.relu,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(args.weight_decay),
                                      scope='Logits', reuse=False)

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        restore_vars = tf.global_variables();
        restore_saver = tf.train.Saver(restore_vars)

        restore_saver.restore(sess, os.path.join(os.path.expanduser(args.model_dir), ckpt_file))

    return pnet, rnet, onet, sess, embeddings, logits, image_batch_placeholder

def load_models_forward_v2(args, Expr_dataset):
    with tf.device('/cpu:0'):
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)

        # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count = {'CPU': 0})) as sess:
        # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True)) as sess:
        # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:


        # Load the model of face detection
        pnet, rnet, onet = align.face_align_mtcnn.load_align_mtcnn(args.align_model_dir)


        ########### load face verif_expression model #############################
        # Load the model of face verification
        print('Model directory: %s' % args.model_dir)
        meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))

        #facenet.load_model(args.model_dir, meta_file, ckpt_file)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        model_dir_exp = os.path.expanduser(args.model_dir)
        saver = tf.train.import_meta_graph(os.path.join(args.model_dir, meta_file))
        saver.restore(sess, os.path.join(model_dir_exp, ckpt_file))

        if Expr_dataset == 'CK+':
            args_model.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            args_model.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            args_model.keep_probability_placeholder = tf.get_default_graph().get_tensor_by_name('keep_probability:0')
            args_model.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
            args_model.logits = tf.get_default_graph().get_tensor_by_name('logits:0')


        if Expr_dataset == 'FER2013':
            args_model.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            args_model.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            args_model.keep_probability_placeholder = tf.get_default_graph().get_tensor_by_name('keep_probability:0')
            args_model.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
            args_model.logits = tf.get_default_graph().get_tensor_by_name('logits:0')
            args_model.phase_train_placeholder_expression = tf.get_default_graph().get_tensor_by_name('phase_train_expression:0')

        return pnet, rnet, onet, sess, args_model



def compare_faces(args, pnet, rnet, onet, sess):

    start_time = time.time()
    align_imgs, bboxes = align.face_align_mtcnn.align_mtcnn(args, pnet, rnet, onet)
    elapsed_time1 = time.time() - start_time
    # print('Face_loop Elapsed time: %fs\n' % (elapsed_time1))
    # print('The probablity of the same person : %f\n' % simi)


    # Get input and output tensors
    #images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    images_placeholder = tf.get_default_graph().get_tensor_by_name("image_batch:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    # Load images
    image_size = images_placeholder.get_shape()[1]
    images = facenet.load_data_im(align_imgs, False, False, image_size)

    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
    try:
        emb_array = sess.run(embeddings, feed_dict=feed_dict)
    except:
        sess.close()
        print("Unexpected error:", sys.exc_info()[0])

    embeddings1 = emb_array[0]
    embeddings2 = emb_array[1]

    # Caculate the distance of embeddings and verification the two face
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 0)
    simi = 1 - spatial.distance.cosine(embeddings1, embeddings2)

    predict_issame = np.less(dist, args.threshold)
    # predict_issame = np.greater(simi, args.threshold)
    elapsed_time = time.time() - start_time


    # print('Image 1: %s\n' % args.img1)
    # print('Image 2: %s\n' % args.img2)
    # print('The same person: %s.......The distance of the two persons: %f | The threshold: %f\n' % (
    # predict_issame, dist, args.threshold))

    elapsed_time2 = elapsed_time - elapsed_time1
    print(
        'Elapsed time: %fs....Detection: %fs....Face verification %fs\n' % (elapsed_time, elapsed_time1, elapsed_time2))

    # img1 = Image.open(args.img1)
    # img2 = Image.open(args.img2)
    # plotbb(img1, bboxes[0, :])
    #
    # plotbb(img2, bboxes[1, :])
    #
    # raw_input('pause...')
    return predict_issame, dist, bboxes

def face_expression(face_img_, img_ref_, args, sess):
    start_time = time.time()
    imgs = np.zeros((2, args.image_size, args.image_size, 3))
    imgs[0, :, :, :]=face_img_
    imgs[1, :, :, :] = img_ref_

    express_probs = []

    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    #images_placeholder = tf.get_default_graph().get_tensor_by_name("image_batch:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    #keep_probability_placeholder = tf.get_default_graph().get_tensor_by_name("keep_probability:0")

    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    #logits = tf.get_default_graph().get_tensor_by_name("logits:0")


    # Load images
    image_size = images_placeholder.get_shape()[1]
    images = facenet.load_data_im(imgs, False, False, image_size)

    #feed_dict = {images_placeholder: images, phase_train_placeholder: False, keep_probability_placeholder:1.0}
    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
    try:
        #emb_array, logits_array = sess.run([embeddings, logits], feed_dict=feed_dict)
        emb_array = sess.run(embeddings, feed_dict=feed_dict)
    except:
        sess.close()
        print("Unexpected error:", sys.exc_info()[0])

    embeddings1 = emb_array[0]
    embeddings2 = emb_array[1]


    # Caculate the distance of embeddings and verification the two face
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 0)
    simi = 1 - spatial.distance.cosine(embeddings1, embeddings2)

    predict_issame = np.less(dist, args.threshold)
    # predict_issame = np.greater(simi, args.threshold)

    # logits0 = logits_array[0]
    # express_probs = np.exp(logits0)/sum(np.exp(logits0))

    elapsed_time = time.time() - start_time


    # print('Image 1: %s\n' % args.img1)
    # print('Image 2: %s\n' % args.img2)
    # print('The same person: %s.......The distance of the two persons: %f | The threshold: %f\n' % (
    # predict_issame, dist, args.threshold))

    # print(
    #     'Elapsed time: %fs....Detection: %fs....Face verification %fs\n' % (elapsed_time, elapsed_time1, elapsed_time2))

    # img1 = Image.open(args.img1)
    # img2 = Image.open(args.img2)
    # plotbb(img1, bboxes[0, :])
    #
    # plotbb(img2, bboxes[1, :])
    #
    # raw_input('pause...')
    return predict_issame, dist, express_probs

def face_expression_multiref(face_img_, img_refs_, args, sess):
    start_time = time.time()
    if len(img_refs_.shape) == 4:
        nrof_imgs = 1+img_refs_.shape[0]
    elif len(img_refs_.shape) == 3:
        nrof_imgs = 2
    else:
        raise ValueError("Dimensions of img_refs is not correct!")

    imgs = np.zeros((nrof_imgs, args.image_size, args.image_size, 3))
    imgs[0, :, :, :]=face_img_
    imgs[1:nrof_imgs, :, :, :] = img_refs_

    express_probs = []

    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    keep_probability_placeholder = tf.get_default_graph().get_tensor_by_name('keep_probability:0')
    # weight_decay_placeholder = tf.get_default_graph().get_tensor_by_name('weight_decay:0')
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
    logits = tf.get_default_graph().get_tensor_by_name('logits:0')

    # images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    # #images_placeholder = tf.get_default_graph().get_tensor_by_name("image_batch:0")
    # phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    # keep_probability_placeholder = tf.get_default_graph().get_tensor_by_name("keep_probability:0")
    #
    # embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    # logits = tf.get_default_graph().get_tensor_by_name("logits:0")


    # Load images
    image_size = images_placeholder.get_shape()[1]
    images = facenet.load_data_im(imgs, False, False, image_size)

    feed_dict = {images_placeholder: images, phase_train_placeholder: False, keep_probability_placeholder:1.0}
    #feed_dict = {images_placeholder: images, phase_train_placeholder: False}
    try:
        t2 = time.time()
        emb_array, logits_array = sess.run([embeddings, logits], feed_dict=feed_dict)
        #emb_array = sess.run(embeddings, feed_dict=feed_dict)
        t3 = time.time()
        print('Embedding calculation FPS:%d' % (int(1 / ((t3 - t2)))))
    except:
        sess.close()
        print("Unexpected error:", sys.exc_info()[0])

    embeddings1 = emb_array[0]
    embeddings2 = emb_array[1:len(emb_array)]


    # Caculate the distance of embeddings and verification the two face
    assert (embeddings1.shape[0] == embeddings2[1].shape[0])
    diff = np.subtract(embeddings1, embeddings2)
    if len(diff.shape)==2:
        dist = np.sum(np.square(diff), 1)
    elif len(diff.shape)==1:
        dist = np.sum(np.square(diff), 0)
    else:
        raise ValueError("Dimension of the embeddings2 is not correct!")

    #simi = 1 - spatial.distance.cosine(embeddings1, embeddings2)

    predict_issame = np.less(dist, args.threshold)
    ##predict_issame = np.greater(simi, args.threshold)

    logits0 = logits_array[0]
    express_probs = np.exp(logits0)/sum(np.exp(logits0))

    return predict_issame, dist, express_probs

#def face_embeddings(img_refs_, args, sess, images_placeholder, embeddings, keep_probability_placeholder, phase_train_placeholder):
def face_embeddings(img_refs_, args, sess, args_model, Expr_dataset):
    start_time = time.time()

    # Load images
    image_size = args.image_size

    images = facenet.load_data_im(img_refs_, False, False, image_size)
    if len(images.shape)==3:
        images = np.expand_dims(images, axis=0)

    if Expr_dataset == 'CK+' or 'FER2013':
        # feed_dict = {images_placeholder: images}
        feed_dict = {args_model.phase_train_placeholder: False, args_model.images_placeholder: images, args_model.keep_probability_placeholder: 1.0}

    t2 = time.time()
    emb_array = sess.run([args_model.embeddings], feed_dict=feed_dict)
    t3 = time.time()
    print('Embedding calculation FPS:%d' % (int(1 / (t3 - t2))))
    t2 = time.time()

    return emb_array



#def face_expression_multiref_forward(face_img_, emb_ref, args, sess, images_placeholder, embeddings, keep_probability_placeholder, phase_train_placeholder, logits):
def face_expression_multiref_forward(face_img_, emb_ref, args, sess, args_model, Expr_dataset):
    start_time = time.time()
    # if len(img_refs_.shape) == 4:
    #     nrof_imgs = 1+img_refs_.shape[0]
    # elif len(img_refs_.shape) == 3:
    #     nrof_imgs = 2
    # else:
    #     raise ValueError("Dimensions of img_refs is not correct!")
    nrof_imgs = 1
    imgs = np.zeros((nrof_imgs, args.image_size, args.image_size, 3))
    imgs[0, :, :, :]=face_img_

    # Load images
    image_size = args.image_size

    images = facenet.load_data_im(imgs, False, False, image_size)
    if len(images.shape) == 3:
        images = np.expand_dims(images,axis=0)

    if Expr_dataset == 'CK+':
        feed_dict = {args_model.phase_train_placeholder: False, args_model.images_placeholder: images, args_model.keep_probability_placeholder: 1.0}
    if Expr_dataset == 'FER2013':
        feed_dict = {args_model.phase_train_placeholder: False, args_model.phase_train_placeholder_expression: False, args_model.images_placeholder: images, args_model.keep_probability_placeholder: 1.0}

    t2 = time.time()
    emb_array, logits_array = sess.run([args_model.embeddings, args_model.logits], feed_dict=feed_dict)
    t3 = time.time()
    print('Embedding calculation FPS:%d' % (int(1 / (t3 - t2))))
    t2 = time.time()
    embeddings1 = emb_array[0]
    embeddings2 = emb_ref[0]


    # Caculate the distance of embeddings and verification the two face
    assert (embeddings1.shape[0] == embeddings2[0].shape[0])
    diff = np.subtract(embeddings1, embeddings2)
    if len(diff.shape)==2:
        dist = np.sum(np.square(diff), 1)
    elif len(diff.shape)==1:
        dist = np.sum(np.square(diff), 0)
    else:
        raise ValueError("Dimension of the embeddings2 is not correct!")

    #simi = 1 - spatial.distance.cosine(embeddings1, embeddings2)

    predict_issame = np.less(dist, args.threshold)
    ##predict_issame = np.greater(simi, args.threshold)

    logits0 = logits_array[0]
    express_probs = np.exp(logits0)/sum(np.exp(logits0))
    return predict_issame, dist, express_probs

def face_verif_batch(face_img_batch, img_refs_, args, sess):
    start_time = time.time()
    predict_issame_batch = []
    dist_batch = []
    if len(img_refs_.shape) == 4:
        nrof_imgs = len(face_img_batch)+img_refs_.shape[0]
    elif len(img_refs_.shape) == 3:
        nrof_imgs = len(face_img_batch) + 1
    else:
        raise ValueError("Dimensions of img_refs is not correct!")

    imgs = np.zeros((nrof_imgs, args.image_size, args.image_size, 3))
    imgs[0:len(face_img_batch), :, :, :]=np.array(face_img_batch)
    imgs[len(face_img_batch):nrof_imgs, :, :, :] = img_refs_

    express_probs = []

    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    #images_placeholder = tf.get_default_graph().get_tensor_by_name("image_batch:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    #keep_probability_placeholder = tf.get_default_graph().get_tensor_by_name("keep_probability:0")

    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    #logits = tf.get_default_graph().get_tensor_by_name("logits:0")


    # Load images
    image_size = images_placeholder.get_shape()[1]
    images = facenet.load_data_im(imgs, False, False, image_size)

    #feed_dict = {images_placeholder: images, phase_train_placeholder: False, keep_probability_placeholder:1.0}
    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
    try:
        t2 = time.time()
        #emb_array, logits_array = sess.run([embeddings, logits], feed_dict=feed_dict)
        emb_array = sess.run(embeddings, feed_dict=feed_dict)
        t3 = time.time()
        print('Embedding calculation FPS:%d' % (int(1 / (t3 - t2) * nrof_imgs)))
    except:
        sess.close()
        print("Unexpected error:", sys.exc_info()[0])

    embeddings1 = emb_array[0:len(face_img_batch)]
    embeddings2 = emb_array[len(face_img_batch):len(emb_array)]


    # Caculate the distance of embeddings and verification the two face
    assert (embeddings1[1].shape[0] == embeddings2[1].shape[0])
    for i in range(nrof_imgs-len(face_img_batch)):
        diff = np.subtract(embeddings1, embeddings2[i])
        if len(diff.shape)==2:
            dist = np.sum(np.square(diff), 1)
        elif len(diff.shape)==1:
            dist = np.sum(np.square(diff), 0)
        else:
            raise ValueError("Dimension of the embeddings2 is not correct!")

        #simi = 1 - spatial.distance.cosine(embeddings1, embeddings2)

        predict_issame = np.less(dist, args.threshold)

        # predict_issame = np.greater(simi, args.threshold)

        # logits0 = logits_array[0]
        # express_probs = np.exp(logits0)/sum(np.exp(logits0))
        predict_issame_batch.append(predict_issame)
        dist_batch.append(dist)
        elapsed_time = time.time() - start_time

    return predict_issame_batch, dist_batch




def plotbb(img, bboxes, ld=[], output_filename=[]):
    #img = np.float32(image)
    # plt.imshow(img)
    # patterns = ['-', '+', 'x', 'o', 'O', '.', '*']  # more patterns


    fig, ax = plt.subplots(1)
    ax.imshow(img)

    if bboxes.ndim <2:
        bboxes = np.expand_dims(bboxes, axis=0)

    for i in range(bboxes.shape[0]):
        rect = patches.Rectangle(
            (bboxes[i, 0], bboxes[i, 1]),
            bboxes[i, 2] - bboxes[i, 0],
            bboxes[i, 3] - bboxes[i, 1],
            # hatch=patterns[i],
            fill=False,
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        score = '%.02f' % (bboxes[i, 4])
        ax.text(int(bboxes[i, 0]), int(bboxes[i, 1]), score, color='green', fontsize=10)
        plt.pause(0.0001)

    if(ld):
        ax = plt.gca()
        ld = np.int32(np.squeeze(ld))
        for i in range(np.int32(ld.shape[0] / 2)):
            ax.plot(ld[i], ld[i + 5], 'o', color='r', linewidth=0.1)

    if (output_filename):
        dirtmp = output_filename
        if not os.path.exists(dirtmp):
            os.mkdir(dirtmp)
        random_key = np.random.randint(0, high=99999)
        fig.savefig(os.path.join(dirtmp, 'face_dd_ld_%03d.png') % random_key, dpi=90, bbox_inches='tight')




class args_model():
    def __init__(self):
        self.images_placeholder = None
        self.embeddings = None
        self.keep_probability_placeholder = None
        self.phase_train_placeholder = None
        self.logits = None
        self.phase_train_placeholder_expression = None