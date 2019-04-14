### FaceLivNet 1.0 : End-to-End Networks based on the Inception-RsNet Combined with face verification and Liveness
### detection based on the facial expression recognition #########
### Author: mingzuheng, 25/12/2017 #########
## papers: FaceLiveNet: End-to-End Networks Combining Face Verification With Interactive Facial Expression-based Liveness Detection
##############################  FACIAL EXPRESSION DATASETS   ##############################
### FER2013(unconstrained/webimages): 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
###                                   Training: 28,709 frames
###                                   PublicTest: 3,589 frames
###                                   PrivateTest: 3,589 frames
###
### CK+ (constrained/posed): 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
###      (327 videos with annotation labels, the last three frames of each video were used as the key frames,
###      and the first frame of every video is the neutral frame and the last frame is the peak frame of expression,
###      1308 frames totally used for training/test, 10-fold cross-validation)
###
### Oulu-CASIA (constrained/posed): 0=Neutral, 1=Anger, 2=Disgust, 3Fear, 4=Happiness, 5=Sadness, 6=Surprise
###             (VIS/Strong: 480 videos = 80 subjects x 6 expressons,
###             every video starts from the neural frame(first frame is neutral). Thus , the first frame and the last three
###             frames (peak frame of expression is the last frame) of each video are used as the expression key frames
###             as the experiment on CK+. 1920 frames totally used for training/test, 10-fold cross-validation)
##############################  FACIAL EXPRESSION DATASETS   ##############################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
#import facenet
#import lfw
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import h5py
from sklearn.decomposition import PCA
import glob
import shutil
import matplotlib.pyplot as plt
import csv
import cv2
import math
import glob
from numpy import linalg as LA
import imp
from tensorflow.python import pywrap_tensorflow




###### user custom lib
import facenet_ext
import lfw_ext
import metrics_loss
import train_BP




# ### CK+  ###
# EXPRSSIONS_TYPE =  ['0=neutral', '1=anger', '2=contempt', '3=disgust', '4=fear', '5=happy', '6=sadness', '7=surprise']
## OULU-CASIA  ###
EXPRSSIONS_TYPE =  ['0=Anger', '1=Disgust', '2=Fear', '3=Happiness', '4=Sadness', '5=Surprise' ]
# ### SFEW  ###
# EXPRSSIONS_TYPE =  ['0=Angry', '1=Disgust', '2=Fear', '3=Happy', '4=Neutral', '5=Sad', '6=Surprise']
### FER2013 ###
#EXPRSSIONS_TYPE =  ['0=Angry', '1=Disgust', '2=Fear', '3=Happy', '4=Sad', '5=Surprise', '6=Neutral']
### FER2013_FUSION ###
#EXPRSSIONS_TYPE = ['0=Neutral', '1=Anger', '2=Disgust', '3=Fear', '4=Happiness', '5=Sadness', '6=Surprise']

def main(args):

    module_networks = str.split(args.model_def, '/')[-1]
    network = imp.load_source(module_networks, args.model_def)
    #network = importlib.import_module(args.model_def, 'inference')
    #network = importlib.import_module(args.model_def)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    random.seed(args.seed)

    #                                                                                               args.labels_expression,
    #                                                                                               'Training', args.augfer2013)

    ###########################   CK+    ##########################
    # image_list, label_list, usage_list, nrof_classes = facenet_ext.get_image_paths_and_labels_ckplus(args.data_dir,
    #                                                                                               args.labels_expression,
    #                                                                                               'Training', args.nfold, args.ifold)

    ##########################   OULU-CASIA    ##########################
    image_list, label_list, usage_list, nrof_classes = facenet_ext.get_image_paths_and_labels_oulucasia(args.data_dir,
                                                                                                args.labels_expression,
                                                                                                'Training', args.nfold,
                                                                                                args.ifold)

    # ############################   SFEW    ##########################
    # image_list, label_list, nrof_classes = facenet_ext.get_image_paths_and_labels_sfew(args.data_dir, args.labels_expression)

    #for img in image_list:
    #	print(img)
    #    strtmp=str.split(img,'/')
    #    img_str = strtmp[-1]
    # 	img_name = str.split(img_str,'.')[0]
    #	im = cv2.imread(img)
    #    #im = cv2.resize(im,(182,182))
    #    cv2.imwrite(os.path.join('/data/zming/datasets/fer2013/raw_182_160_png',img_name+'.png'),im)
      


    print('Total number of subjects: %d' % nrof_classes)
    print('Total number of images: %d' % len(image_list))
    # if args.filter_filename:
    #     print('Filtering...')
    #     train_set = filter_dataset(train_set, args.filter_filename,
    #         args.filter_percentile, args.filter_min_nrof_images_per_class, args.trainset_start)

    #nrof_classes = len(train_set)
    # Get a list of image paths and their labels
    #image_list, label_list = facenet.get_image_paths_and_labels(train_set)
    #image_list, label_list = facenet.get_image_paths_and_labels_expression(train_set, args.labels_expression)
    # print('Total number of subjects: %d' % nrof_classes)
    # print('Total number of images: %d' % len(image_list))
    
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
        pairs = lfw_ext.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, actual_issame = lfw_ext.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)

    if args.evaluate_express:
        print('Test data directory: %s' % args.data_dir)
        ###########################   FER2013    ##########################
        # image_paths_test, label_list_test, usage_list_test, nrof_classes_test = facenet_ext.get_image_paths_and_labels_fer2013(args.data_dir,
        #                                                                                               args.labels_expression,
        #                                                                                               'PublicTest')

        ###########################   CK+    ##########################
        # image_paths_test, label_list_test, usage_list_test, nrof_classes_test = facenet_ext.get_image_paths_and_labels_ckplus(
        #     args.data_dir, args.labels_expression, 'Test', args.nfold, args.ifold)

        ##########################   OULU-CASIA    ##########################
        image_paths_test, label_list_test, usage_list_test, nrof_classes_test = facenet_ext.get_image_paths_and_labels_oulucasia(
           args.data_dir, args.labels_expression, 'Test', args.nfold, args.ifold)
        
        # ###########################   SFEW    ##########################
        # image_paths_test, label_list_test, nrof_classes_test = facenet_ext.get_image_paths_and_labels_sfew(args.data_dir_test, args.labels_expression_test)


        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)
        

        
        # Create a queue that produces indices into the image_list and label_list 
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                             shuffle=True, seed=None, capacity=32)
        
        index_dequeue_op = index_queue.dequeue_many(args.batch_size*args.epoch_size, 'index_dequeue')
        
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        phase_train_placeholder_expression = tf.placeholder(tf.bool, name='phase_train_expression')

        
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')

        labels_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='labels')

        keep_probability_placeholder = tf.placeholder(tf.float32, name='keep_probability')

        input_queue = data_flow_ops.FIFOQueue(max(100000, args.batch_size*args.epoch_size),
                                    dtypes=[tf.string, tf.int64],
                                    shapes=[(1,), (1,)],
                                    shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder], name='enqueue_op')
        
        nrof_preprocess_threads = 4
        #nrof_preprocess_threads = 1
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            #for filename in tf.unpack(filenames): ##tf 012
            for filename in tf.unstack(filenames):  ##tf100
                file_contents = tf.read_file(filename)
                image = tf.image.decode_png(file_contents)
                #image = tf.image.decode_jpeg(file_contents)
                if args.random_rotate:
                    image = tf.py_func(facenet.random_rotate_image, [image], tf.uint8)
                if args.random_crop:
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    #image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                    image = tf.image.resize_images(image, [args.image_size, args.image_size]) ## if input is face image, keep the whole image
                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)
    
                #pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])
    
        image_batch, label_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder, 
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        #image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')
        

        
        print('Building training graph')
        
        # Build the inference graph
        prelogits, end_points = network.inference(image_batch, keep_probability_placeholder,
            phase_train=phase_train_placeholder, weight_decay=args.weight_decay)

        # ### transfer the output of the prelogits to 1600 elements which can respahe to 40x40 as an image input of the
        # ### expression net
        # logits_deconv = slim.fully_connected(prelogits, 1600, activation_fn=None,
        #         weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        #         weights_regularizer=slim.l2_regularizer(args.weight_decay),
        #         scope='logits_deconv', reuse=False)
        # ### reshape the prelogits to the 2D images [batch, width, height]
        # prelogits_deconv = tf.reshape(logits_deconv, [batch_size_placeholder, 40,40,1], name='prelogits_reshape')

        ### the expression net for expression classification, and the input is the reshaped logits_deconv
        #inputs = end_points['MaxPool_3a_3x3']
        #inputs = end_points['Conv2d_2b_3x3']
        #inputs = end_points['Mixed_6a']
        #inputs = end_points['Mixed_5a']
        #inputs = end_points['Conv2d_4b_3x3']
        #inputs = image_batch
        inputs = end_points['Mixed_7a']
        #inputs = end_points['Mixed_8a']
        #inputs = end_points['Mixed_6a']
        #inputs = end_points['Mixed_6.5a']
        prelogits_expression, end_points_expression = network.inference_expression(inputs, keep_probability_placeholder, phase_train=phase_train_placeholder_expression, weight_decay=args.weight_decay)

        logits_0 = slim.fully_connected(prelogits_expression, 128, activation_fn=tf.nn.relu,weights_initializer=tf.truncated_normal_initializer(stddev=0.1), weights_regularizer=slim.l2_regularizer(args.weight_decay),scope='Logits_0', reuse=False)

        #logits_0 = slim.dropout(logits_0, keep_probability_placeholder, is_training=True, scope='Dropout')

        logits = slim.fully_connected(logits_0, len(set(label_list)), activation_fn=tf.nn.relu,weights_initializer=tf.truncated_normal_initializer(stddev=0.1), weights_regularizer=slim.l2_regularizer(args.weight_decay),scope='Logits', reuse=False)

        logits = tf.identity(logits, 'logits')

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Add center loss
        if args.center_loss_factor>0.0:
            #prelogits_center_loss, centers, _, centers_cts_batch_reshape = facenet.center_loss(prelogits, label_batch, args.center_loss_alfa, nrof_classes)
            prelogits_center_loss, _, centers, _, centers_cts_batch_reshape, _ = metrics_loss.center_loss(embeddings, label_batch,
                                                                                               args.center_loss_alfa,
                                                                                               nrof_classes)
            #prelogits_center_loss, _ = facenet.center_loss_similarity(prelogits, label_batch, args.center_loss_alfa, nrof_classes) ####Similarity cosine distance, center loss
            #tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)

        ###### Automatic lower learning rate lr= lr_start * decay_factor^(global_step/decaystepwidth),
        ###### if decay_factor = 1.0 it means the learning rate will not decay automaically, otherwise it will decay
        ###### from the given learning rate in function of the factor, current global step and decay ste width.
        if args.learning_rate>0.0:
            learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                       args.learning_rate_decay_epochs*args.epoch_size,
                                                       args.learning_rate_decay_factor, staircase=True)
        else:
            learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                       args.learning_rate_decay_epochs * args.epoch_size,
                                                       1.0, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     logits, label_batch, name='cross_entropy_per_example') ##tf012
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=label_batch, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean) ##tf100
        
        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')
        total_loss = tf.add_n([cross_entropy_mean], name='total_loss')

        #### Training accuracy of softmax: check the underfitting or overfiting #############################
        correct_prediction = tf.equal(tf.argmax(logits, 1), label_batch)
        softmax_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        ########################################################################################################

        ########## edit mzh   #####################
        # Create list with variables to restore
        restore_vars = []
        restore_vars_expression = []
        update_gradient_vars = []
        
        #for var in tf.global_variables():
        for var in tf.trainable_variables():
            if 'InceptionResnetV1_expression/' in var.op.name or 'Logits/' in var.op.name or 'Logits_0/' in var.op.name:
                # print(var.op.name)
                update_gradient_vars.append(var)
        print('########## Update variables : ')

        parasize = 0
        paracnt = 0
        for var in update_gradient_vars:
            print(var)
            paranum = 1
            for dim in var.get_shape():
                paranum *= dim.value
            parasize += paranum* sys.getsizeof(var.dtype)
            paracnt += paranum
        print('The number of the update parameters in the model is %dM, ......the size is : %dM bytes'%(paracnt/1e6, parasize/1e6))
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        train_op,grads, grads_clip = train_BP.train(total_loss, global_step, args.optimizer,
            learning_rate, args.moving_average_decay, update_gradient_vars, summary_op, args.log_histograms)
        


        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)



        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        #sess = tf.Session(config=tf.ConfigProto(device_count={'CPU':1}, log_device_placement=True))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        tf.train.start_queue_runners(sess=sess) ## wakeup the queue: start the queue operating defined by the tf.train.batch_join

        with sess.as_default():

            if pretrained_model:
                reader = pywrap_tensorflow.NewCheckpointReader(os.path.join(os.path.expanduser(args.pretrained_model), ckpt_file))
                var_to_shape_map = reader.get_variable_to_shape_map()
                i=0
                isExpressionModel = False
                for key in sorted(var_to_shape_map):
                    print(key)
                    ##xx= reader.get_tensor(key)
                    if 'InceptionResnetV1_expression' in key:
                        isExpressionModel = True
                    i += 1
                print('################## %d parametrs in the pretrained model ###################'%i)

                if isExpressionModel:
                    print('>>>>>>>>>>>> Loading directly the pretrained FaceLiveNet model :%s.....'% os.path.join(os.path.expanduser(args.pretrained_model), ckpt_file))
                    #restore_vars = tf.trainable_variables()
                    #for var in tf.trainable_variables():
                    for var in tf.global_variables():
                        if 'center' not in var.op.name:
                            restore_vars.append(var)

                    parasize = 0
                    paracnt = 0
                    for var in restore_vars:
                        print(var)
                        paranum = 1
                        for dim in var.get_shape():
                            paranum *= dim.value
                        parasize += paranum * sys.getsizeof(var.dtype)
                        paracnt += paranum
                    print('The number of the loading parameters in the model(FaceLiveNet) is %dM, ......the size is : %dM bytes' % (
                        paracnt / 1e6, parasize / 1e6))
                    restore_saver_expression = tf.train.Saver(restore_vars)
                    restore_saver_expression.restore(sess, os.path.join(os.path.expanduser(args.pretrained_model), ckpt_file))
                else:
                    print('Restoring pretrained model: %s' % os.path.join(os.path.expanduser(args.pretrained_model), ckpt_file))
                    #saver.restore(sess, pretrained_model)
                    for var in tf.global_variables():
                        if 'InceptionResnetV1/' in var.op.name:
                            restore_vars.append(var)
                        if 'InceptionResnetV1_expression/' in var.op.name:
                            restore_vars_expression.append(var)

                    parasize = 0
                    paracnt = 0
                    for var in restore_vars:
                        # print(var)
                        paranum = 1
                        for dim in var.get_shape():
                            paranum *= dim.value
                        parasize += paranum * sys.getsizeof(var.dtype)
                        paracnt += paranum
                    print(
                        'The number of the loading parameters in the model(Facenet) is %dM, ......the size is : %dM bytes' % (
                            paracnt / 1e6, parasize / 1e6))

                    restore_saver=tf.train.Saver(restore_vars)
                    restore_saver.restore(sess, os.path.join(os.path.expanduser(args.pretrained_model), ckpt_file))
                    j = 0
                    for i, var in enumerate(restore_vars_expression):
                        var_name = str.split(str(var.op.name), 'InceptionResnetV1_expression/')[1]
                        #### only used for the block8 branch cut case ##########
                        #if 'block8_branchcut_1/Conv2d_1x1' in var_name:
                        #	continue
                        #### only used for the block8 branch cut case ##########
                        if 'Repeat' in var_name:
                            var_name = str.split(var_name,'Repeat')[1]
                            pos = var_name.index('/')
                            var_name = var_name[pos:]
                            if 'block8_branchcut_1/' in var_name:
                                var_name = str.split(var_name,'block8_branchcut_1/')[1]
                                var_name = 'block8_1/'+var_name
                                #var_name = 'block8_1'+var_name[1:]
                                #var_name = 'block8_'+var_name
                        for var0 in restore_vars:
                            if var_name in str(var0.op.name):
                                sess.run(var.assign(var0))
                                print(j)
                                var0_sum = np.sum(sess.run(var0))
                                var_sum = np.sum(sess.run(var))
                                print(var0.op.name, '===========>>>>>>>>> var0_sum:%f'%var0_sum)
                                print(var.op.name, '===========>>>>>>>>>  var_sum:%f' %var_sum)
                                if var0_sum != var_sum:
                                    raise ValueError('Error of the assignment form var0 to var!')
                                j += 1
                                break



        # Training and validation loop
            print('Running training')
            epoch = 0
            acc = 0
            val = 0
            far = 0
            best_acc = 0
            acc_expression = 0
            nrof_expressions = len(set(label_list))
            each_expr_acc = np.zeros(nrof_expressions)
            express_probs_confus_matrix = np.zeros((nrof_expressions,nrof_expressions))
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                print('Epoch step: %d'%step)
                epoch = step // args.epoch_size
                # Train for one epoch
                step, train_each_expr_acc, softmax_acc_, grads_total_sum, loss, cross_entropy_mean_, Reg_loss,  center_loss_, verifacc, learning_rate_ = train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
                    learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step, 
                    total_loss, train_op, summary_op, summary_writer, regularization_losses, args.learning_rate_schedule_file,
                      prelogits_center_loss, cross_entropy_mean, acc, val, far, centers_cts_batch_reshape, softmax_acc,
                      logits, keep_probability_placeholder, restore_vars, update_gradient_vars, acc_expression,
                      each_expr_acc, label_batch, express_probs_confus_matrix, log_dir, model_dir, grads, grads_clip,
                      cross_entropy, image_batch, learning_rate, phase_train_placeholder_expression, best_acc)

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

                ## Evaluate on LFW
                if not (epoch%100):
                     if args.lfw_dir:
                         acc, val, far = evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder,
                                                  phase_train_placeholder, batch_size_placeholder,embeddings, label_batch,
                                                  lfw_paths, actual_issame, args.lfw_batch_size, args.lfw_nrof_folds,
                                                  log_dir, step, summary_writer, args.evaluate_mode,
                                                  keep_probability_placeholder)

                ## Evaluate for expression classification
                if not (epoch % 1 ):
                    if args.evaluate_express:
                        acc_expression, each_expr_acc, exp_cnt, expredict_cnt,  express_probs_confus_matrix, express_recog_images = evaluate_expression(sess, enqueue_op, image_paths_placeholder, labels_placeholder, batch_size_placeholder, logits, label_batch, image_paths_test, label_list_test, args.lfw_batch_size, log_dir, step, summary_writer, keep_probability_placeholder,input_queue, phase_train_placeholder_expression, phase_train_placeholder)


                ## saving the best_model
                if acc_expression>best_acc:
                    best_acc = acc_expression
                    best_model_dir = os.path.join(model_dir, 'best_model')
                    if not os.path.isdir(best_model_dir):  # Create the log directory if it doesn't exist
                        os.makedirs(best_model_dir)
                    if os.listdir(best_model_dir):
                        for file in glob.glob(os.path.join(best_model_dir, '*.*')):
                            os.remove(file)
                    for file in glob.glob(os.path.join(model_dir, '*.*')):
                        shutil.copy(file, best_model_dir)

                    ######################## SAVING BEST CONFUSION RESULTS IMAGES  ################################
                    images_exists = glob.glob(os.path.join(best_model_dir, 'confusion_matrix_images*'))
                    for folder in images_exists:
                        shutil.rmtree(folder)

                    confus_images_folder = os.path.join(best_model_dir, 'confusion_matrix_images_%dsteps' % step)

                    os.mkdir(confus_images_folder)

                    ## mkdir for the confusion matrix of each expression
                    for i in range(nrof_expressions):
                        gt_folder = os.path.join(confus_images_folder, '%d') % i
                        os.mkdir(gt_folder)
                        for j in range(nrof_expressions):
                            predict_folder = os.path.join(confus_images_folder, '%d', '%d') % (i, j)
                            os.mkdir(predict_folder)

                    ## copy the predicting images to the corresponding folder of the predicting expression
                    for i, labs_predict in enumerate(express_recog_images):
                        for j, lab_predict in enumerate(labs_predict):
                            dst = os.path.join(confus_images_folder, '%d', '%d') % (i, j)
                            for img in express_recog_images[i][j]:
                                shutil.copy(img[0], dst)
                    ######################## SAVING BEST CONFUSION RESULTS IMAGES  ################################

                ###################   Saving the confusion matrix  ##########################################
                with open(os.path.join(log_dir, 'confusion_matrix.txt'), 'a') as f:
                    f.write('%d expressions recognition TRAINING accuracy is: %2.3f\n' % (nrof_expressions, softmax_acc_))
                    f.write('loss: %2.3f  crossentropy: %2.3f  regloss: %2.3f  centerloss: %2.3f  verifAcc: %2.3f  lr:%e \n' % (loss, cross_entropy_mean_, Reg_loss,  center_loss_, verifacc, learning_rate_))
                    line = ''
                    for idx, expr in enumerate(EXPRSSIONS_TYPE):
                        line += (expr + ': %2.3f,  ')%train_each_expr_acc[idx]
                    f.write('Training acc: '+line + '\n')

                    f.write('%d expressions recognition TEST accuracy is: %2.3f\n' % (nrof_expressions, acc_expression))
                    f.write('>>>>>>>>>>>>>>>>>>>>>>>>>>> Gradient norm**2 is: %f\n' % grads_total_sum)
                    f.write('--------  Confusion matrix expressions recog AFTER %d steps of the iteration: ---------------\n' % step)
                    line = ''
                    for expr in EXPRSSIONS_TYPE:
                        line += expr + ',  '
                    f.write(line + '\n')

                    for i in range(nrof_expressions):
                        line = ''
                        line += '%d   ' % i
                        for j in range(nrof_expressions):
                            line += '%2.3f ' % express_probs_confus_matrix[i][j]
                        f.write(line + '\n')
                    f.write('----------------------------------------------------------------------------------------\n')
                ###################   Saving the confusion matrix  ##########################################

    return model_dir
  
def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    #plt.plot(bin_centers, cdf)
    threshold = np.interp(percentile*0.01, cdf, bin_centers)
    return threshold
  
def filter_dataset(dataset, data_filename, percentile, min_nrof_images_per_class,trainset_start):
    with h5py.File(data_filename,'r') as f:
        distance_to_center = np.array(f.get('distance_to_center'))
        label_list = np.array(f.get('label_list'))
        image_list = np.array(f.get('image_list'))
        distance_to_center_threshold = find_threshold(distance_to_center, percentile)
        indices = np.where(distance_to_center>=distance_to_center_threshold)[0]
        filtered_dataset = dataset
        removelist = []
        for i in indices:
            label = label_list[i]
            image = image_list[i]
            if label >= trainset_start:
                if image in filtered_dataset[label-trainset_start].image_paths:
                    filtered_dataset[label-trainset_start].image_paths.remove(image)
                if len(filtered_dataset[label-trainset_start].image_paths)<min_nrof_images_per_class:
                    removelist.append(label-trainset_start)


        ix = sorted(list(set(removelist)), reverse=True)
        for i in ix:
            del(filtered_dataset[i])

    return filtered_dataset
  
def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder, 
      learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step, 
      loss, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file, prelogits_center_loss,
          cross_entropy_mean, acc, val, far, centers_cts_batch_reshape, softmax_acc, logits, keep_probability_placeholder,
          restore_vars, update_gradient_vars, acc_expression, each_expr_acc, label_batch, express_probs_confus_matrix,
          log_dir, model_dir, grads, grads_clip, cross_entropy, image_batch, learning_rate, phase_train_placeholder_expression, best_acc):
    batch_number = 0
    
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    print('Index_dequeue_op....')
    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]
    for i in range(10):
        print('index: %d' %index_epoch[i])
        print('label_epoch: %d image_epoch: %s' % (label_epoch[i], image_epoch[i]))



    
    print('Enqueue__op....')
    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch),1)
    image_paths_array = np.expand_dims(np.array(image_epoch),1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

    # Training loop
    train_time = 0

    ####################### summing up the values the dimensions of the variables for checking the updating of the variables ##########################
    vars_ref = []
    check_vars = update_gradient_vars
    #check_vars = restore_vars
    #check_vars = tf.trainable_variables()
    for var in check_vars:
    #for var in tf.trainable_variables():
        var_value = sess.run(var)
        vars_ref.append(np.sum(var_value))
    ####################### summing up the values the dimensions of the variables for checking the updating of the variables ##########################
    # Add validation loss and accuracy to summary
    summary = tf.Summary()

    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: False, phase_train_placeholder_expression: True, batch_size_placeholder: args.batch_size, keep_probability_placeholder: args.keep_probability}
        if (batch_number % 100 == 0):
            err, _, step, reg_loss, prelogits_center_loss_, cross_entropy_mean_, centers_cts_batch_reshape_, softmax_acc_, logits_, label_batch_, grads_, grads_clip_, cross_entropy_, image_batch_, learning_rate_, summary_str = sess.run([loss, train_op, global_step, regularization_losses, prelogits_center_loss, cross_entropy_mean, centers_cts_batch_reshape, softmax_acc, logits, label_batch, grads, grads_clip, cross_entropy, image_batch, learning_rate, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            err, _, step, reg_loss, prelogits_center_loss_, cross_entropy_mean_, centers_cts_batch_reshape_, softmax_acc_, logits_, label_batch_, grads_, grads_clip_, cross_entropy_, image_batch_, learning_rate_ = sess.run([loss, train_op, global_step, regularization_losses,prelogits_center_loss, cross_entropy_mean, centers_cts_batch_reshape, softmax_acc, logits, label_batch, grads, grads_clip, cross_entropy, image_batch, learning_rate], feed_dict=feed_dict)
        print("step %d"%step)
        duration = time.time() - start_time



        ####################### check the state of the update weights/bias in the variables by summing up the values the dimensions of the variables ##########################
        vars_sum = 0
        ii = 0
        dd = 0
        parasize = 0
        paracnt = 0
        for var in check_vars:
            var_value = sess.run(var)
            if np.sum(var_value) != vars_ref[ii]:
                #print(var.op.name)
                dd += 1
                #raw_input()
            vars_sum += np.sum(var_value)
            ii +=1

            paranum = 1
            for dim in var.get_shape():
                paranum *= dim.value
            parasize += paranum * sys.getsizeof(var.dtype)
            paracnt += paranum
        print('%d vars changed'%dd)
        print('The number of the update parameters in the model is %dM, ......the size is : %dM bytes' % (paracnt / 1e6, parasize / 1e6))
        ####################### check the state of the update weights/bias in the variables by summing up the values the dimensions of the variables ##########################


        ####### gradients values checking ####################
        grads_sum = 0
        grad_clip_sum = 0
        ### grad[0] is the gradient, and the grad[1] is value of the variable(i.e. weights/bias...)
        for i, grad in enumerate(grads_):
            grad_norm = LA.norm(np.asarray(grad[0]))
            # if math.isnan(grad_norm):
            #     print(grad)
            grads_sum += grad_norm**2
            print ('grad_%dth: %f  '%(i,grad_norm), end='')
        print('\n')
        for i, grad_clip in enumerate(grads_clip_):
            grad_clip_norm = LA.norm(np.asarray(grad_clip))
            grad_clip_sum += grad_clip_norm**2
            print ('grad_clip_%dth: %f  '%(i,grad_clip_norm), end='')
        print('\n')
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>> Gradient norm is: %f' % math.sqrt(grads_sum))
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>> Gradient clip norm is: %f' % math.sqrt(grad_clip_sum))



        ############# the accuracy of each expression  ################
        express_probs = np.exp(logits_) / np.tile(
            np.reshape(np.sum(np.exp(logits_), 1), (logits_.shape[0], 1)), (1, logits_.shape[1]))
        nrof_expression = len(set(label_list))
        expressions_predict = np.argmax(express_probs, 1)

        exp_cnt = np.zeros(nrof_expression)
        expredict_cnt = np.zeros(nrof_expression)
        for i in range(label_batch_.shape[0]):
            lab = label_batch_[i]
            exp_cnt[lab] += 1
            if lab == expressions_predict[i]:
                expredict_cnt[lab] += 1
        train_each_expr_acc = expredict_cnt / exp_cnt
        ###############################################
        print('########## log dir is: %s '%log_dir )
        print('########## model dir is: %s ' %model_dir)
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tCrossEntropy %2.3f\tRegLoss %2.3f\tCenterLoss_l2 %2.3f\tAcc %2.3f\tVal %2.3f\tFar %2.3f\tsoftmax_acc %2.3f\ttest_expr_acc %2.3f\t lr %e\tbest_acc %2.3f' %
            (epoch, batch_number + 1, args.epoch_size, duration, err, cross_entropy_mean_, np.sum(reg_loss), prelogits_center_loss_, acc, val, far, softmax_acc_, acc_expression, learning_rate_, best_acc))
        
        print('Training each_expression_acc: ',end = '')
        for i, expr in enumerate(EXPRSSIONS_TYPE):
            print(expr+'  %2.3f  '%(train_each_expr_acc[i]),end='')
        print('\n')
       
        print('Test each_expression_acc: ',end = '')
        for i, expr in enumerate(EXPRSSIONS_TYPE):      
            print(expr+'  %2.3f  '%(each_expr_acc[i]),end='') 
        print('\n')
       
        print('---------------------- Confusion matrix expressions recog ----------------------\n')
        for expr in EXPRSSIONS_TYPE:
            print(expr+',  ',end='')
        print('\n')  

        for i in range(nrof_expression):
            print ('%d   '%i, end='')
            for j in range(nrof_expression):
                print ('%2.3f '%express_probs_confus_matrix[i][j], end='')
            print('\n')
        print('----------------------------------------------------------------------------------------\n')


        batch_number += 1
        train_time += duration

        summary.value.add(tag='gradient/grad_total_norm', simple_value=grad_clip_sum)
        summary_writer.add_summary(summary, step)

    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    return step, train_each_expr_acc, softmax_acc_, grad_clip_sum, err, cross_entropy_mean_, np.sum(reg_loss),  prelogits_center_loss_, acc, learning_rate_

def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, 
        embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, log_dir, step, summary_writer,
             evaluate_mode, keep_probability_placeholder):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Running forward pass on LFW images')
    nrof_images = len(actual_issame) * 2
    nrof_batches = nrof_images // batch_size ##floor division

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.arange(0,len(image_paths)),1)
    image_paths_array = np.expand_dims(np.array(image_paths),1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    
    embedding_size = embeddings.get_shape()[1]

    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for _ in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size, keep_probability_placeholder: 1.0}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab] = emb
        
    assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    if evaluate_mode == 'Euclidian':
        _, _, accuracy, val, val_std, far, fp_idx, fn_idx,best_threshold, val_threshold = lfw_ext.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)
    if evaluate_mode == 'similarity':
        pca = PCA(n_components=128)
        pca.fit(emb_array)
        emb_array_pca = pca.transform(emb_array)
        _, _, accuracy, val, val_std, far, fp_idx, fn_idx,best_threshold, val_threshold = lfw_ext.evaluate_cosine(emb_array_pca, actual_issame, nrof_folds=nrof_folds)

    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir,'lfw_result.txt'),'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))

    acc = np.mean(accuracy)
    return acc, val, far

def evaluate_expression(sess, enqueue_op, image_paths_placeholder, labels_placeholder,
             batch_size_placeholder,
             logits, label_batch, image_paths, actual_expre, batch_size, log_dir, step, summary_writer,
             keep_probability_placeholder,input_queue,phase_train_placeholder_expression, phase_train_placeholder):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    #print('Runnning forward pass on FER2013 images')
    print('Running forward pass on expression images')
    nrof_images = len(actual_expre)

    #batch_size = 128

    ############## Enqueue complete batches ##############################
    nrof_batches = nrof_images // batch_size ## The floor division to get the maximum number of the complete batch
    nrof_enqueue = nrof_batches * batch_size

    ############## Allow enqueue incomplete batch  ##############################
    # nrof_batches = int(math.ceil(nrof_images / batch_size)) ## To get the left elements in the queue when the nrof_images can be not exact divided by batch_size
    # nrof_enqueue = nrof_images

    # Enqueue one epoch of image paths and labels
    #labels_array = np.expand_dims(actual_expre[0:nrof_enqueue], 1)
    labels_array = np.expand_dims(np.arange(nrof_enqueue),1)  ## labels_array is not the label of expression of the image, it is the number of the image in the queue
    image_paths_array = np.expand_dims(np.array(image_paths[0:nrof_enqueue]), 1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    #filenames, label = sess.run(input_queue.dequeue())
    logits_size = logits.get_shape()[1]


    logits_array = np.zeros((nrof_enqueue, logits_size), dtype=float)
    ## label_batch_array is not the label of expression of the image , it is the number of the image in the queue.
    ## label_batch_array is used for keeping the order of the labels and images after the batch_join operation which
    ## generates the batch in multi-thread scrambling the order
    label_batch_array = np.zeros(nrof_enqueue, dtype=int)

    for ii in range(nrof_batches):
        print('nrof_batches %d'%ii)
        feed_dict = {phase_train_placeholder: False, phase_train_placeholder_expression: False, batch_size_placeholder: batch_size,
                     keep_probability_placeholder: 1.0}
        ### Capture the exceptions when the queue is exhausted for producing the batch
        try:
            logits_batch, lab = sess.run([logits, label_batch], feed_dict=feed_dict)
        except tf.errors.OutOfRangeError:
            print('Exceptions: the queue is exhausted !')

        label_batch_array[lab] = lab
        logits_array[lab] = logits_batch
    assert np.array_equal(label_batch_array, np.arange(nrof_enqueue)) == True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'

    actual_expre_batch = actual_expre[0:nrof_enqueue]
    express_probs = np.exp(logits_array) / np.tile(np.reshape(np.sum(np.exp(logits_array), 1), (logits_array.shape[0], 1)), (1, logits_array.shape[1]))
    nrof_expression = express_probs.shape[1]
    expressions_predict = np.argmax(express_probs, 1)
    #### Training accuracy of softmax: check the underfitting or overfiting #############################
    correct_prediction = np.equal(expressions_predict, actual_expre_batch)
    test_expr_acc = np.mean(correct_prediction)

    ############# the accuracy of each expression  ################
    ### Initializing the confusion matrix
    exp_cnt = np.zeros(nrof_expression)
    expredict_cnt = np.zeros(nrof_expression)
    express_probs_confus_matrix = np.zeros((nrof_expression, nrof_expression))
    express_recog_images = []
    for i in range(nrof_expression):
        express_recog_images.append([])
        for _ in range(nrof_expression):
            express_recog_images[i].append([])

    ### Fill the confusion matrix
    for i in range(label_batch_array.shape[0]):
        lab  = actual_expre_batch[i]
        exp_cnt[lab] += 1
        express_probs_confus_matrix[lab, expressions_predict[i]] += 1
        express_recog_images[lab][expressions_predict[i]].append(image_paths_array[i])
        if  lab == expressions_predict[i]:
            expredict_cnt[lab] += 1
    test_each_expr_acc = expredict_cnt/exp_cnt
    express_probs_confus_matrix /= np.expand_dims(exp_cnt,1)
    ###############################################

    print('%d expressions recognition accuracy is: %f' % (nrof_expression, test_expr_acc))

    ############### Saving recognition CONFUSION Results images of the 7 expressions  #####################
    print('Saving expression recognition images corresponding to the confusion matrix in %s...'%log_dir)

    images_exists = glob.glob(os.path.join(log_dir, 'confusion_matrix_images*'))
    for folder in images_exists:
        shutil.rmtree(folder)

    confus_images_folder = os.path.join(log_dir, 'confusion_matrix_images_%dsteps' % step)

    os.mkdir(confus_images_folder)

    ## mkdir for the confusion matrix of each expression
    for i in range(nrof_expression):
        gt_folder = os.path.join(confus_images_folder, '%d')%i
        os.mkdir(gt_folder)
        for j in range(nrof_expression):
            predict_folder = os.path.join(confus_images_folder, '%d', '%d')%(i,j)
            os.mkdir(predict_folder)

    ## copy the predicting images to the corresponding folder of the predicting expression
    for i, labs_predict in enumerate(express_recog_images):
        for j, lab_predict in enumerate(labs_predict):
            dst = os.path.join(confus_images_folder, '%d', '%d')%(i,j)
            for img in express_recog_images[i][j]:
                shutil.copy(img[0], dst)
    ############### Saving recognition results images of the 7 expressions  #####################

    fer_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='fer/accuracy', simple_value=test_expr_acc)
    summary.value.add(tag='time/fer', simple_value=fer_time)
    summary_writer.add_summary(summary, step)
    #with open(os.path.join(log_dir, 'Fer2013_result.txt'), 'at') as f:
    with open(os.path.join(log_dir, 'Expression_result.txt'), 'at') as f:
        f.write('%d\t%.5f\n' % (step, test_expr_acc))


    return test_expr_acc, test_each_expr_acc, exp_cnt, expredict_cnt, express_probs_confus_matrix, express_recog_images


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
  

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        default='/data/zming/datasets/CK+/CK+_mtcnnpy_182_160_expression')
    parser.add_argument('--data_dir_test', type=str,
        help='Path to the data directory containing aligned face for test. Multiple directories are separated with colon.')
    parser.add_argument('--labels_expression', type=str,
        help='Path to the Emotion labels file.', default='~/datasets/zming/datasets/CK+/Emotion_labels.txt')
    parser.add_argument('--labels_expression_test', type=str,
        help='Path to the Emotion labels file for test.', default='~/datasets/zming/datasets/CK+/Emotion_labels.txt')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.nn4')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=96)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate', 
        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--decov_loss_factor', type=float,
        help='DeCov loss factor.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,
        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--optimizer', type=str, choices=['Adagrad', 'Adadelta', 'Adam', 'RMSProp', 'Momentum'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=10)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=0.1)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
        help='Number of preprocessing (data loading and augumentation) threads.', default=4)
    parser.add_argument('--log_histograms', 
        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='../data/learning_rate_schedule.txt')
    parser.add_argument('--filter_filename', type=str,
        help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--filter_percentile', type=float,
        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
        help='Keep only the classes with this number of examples or more', default=0)
    parser.add_argument('--evaluate_express', type=bool,
                        help='Whether having the valdating process during the training', default=True)
    parser.add_argument('--augfer2013',
                       help='Whether doing the augmentation for the FER2013 dataset to balance the classes', action='store_true')
    parser.add_argument('--nfold', type=int,
                        help='The ith fold used in the n-fold cross-validation', default=10)
    parser.add_argument('--ifold', type=int,
                        help='The ith fold used in the n-fold cross-validation', default=0)


    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.', default='../data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
                        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_dir', type=str,
                        help='Path to the data directory containing aligned face patches.')
    parser.add_argument('--lfw_batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--trainset_start', type=int,
        help='Number of the start of the train set', default=0)
    parser.add_argument('--trainset_end', type=int,
        help='Number of the end of the train set')
    parser.add_argument('--evaluate_mode', type=str,
                        help='The evaluation mode: Euclidian distance or similarity by cosine distance.',
                        default='Euclidian')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
