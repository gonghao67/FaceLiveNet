# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from subprocess import Popen, PIPE
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from scipy import interpolate
from tensorflow.python.training import training
import random
import re
from collections import Counter
import matplotlib.pyplot as plt
import cv2
#import python_getdents
from scipy import spatial
from sklearn.decomposition import PCA
from itertools import islice
import itertools


def shuffle_examples(image_paths, labels):
    shuffle_list = list(zip(image_paths, labels))
    random.shuffle(shuffle_list)
    image_paths_shuff, labels_shuff = zip(*shuffle_list)
    return image_paths_shuff, labels_shuff

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)
    return example, label
  
def random_rotate_image(image):
    #angle = np.random.uniform(low=-10.0, high=10.0)
    angle = np.random.uniform(low=-180.0, high=180.0)
    return misc.imrotate(image, angle, 'bicubic')
  
def read_and_augument_data(image_list, label_list, image_size, batch_size, max_nrof_epochs, 
        random_crop, random_flip, random_rotate, nrof_preprocess_threads, shuffle=True):
    
    images = ops.convert_to_tensor(image_list, dtype=tf.string)
    labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
    
    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
        num_epochs=max_nrof_epochs, shuffle=shuffle)

    images_and_labels = []
    for _ in range(nrof_preprocess_threads):
        image, label = read_images_from_disk(input_queue)
        if random_rotate:
            image = tf.py_func(random_rotate_image, [image], tf.uint8)
        if random_crop:
            image = tf.random_crop(image, [image_size, image_size, 3])
        else:
            image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        if random_flip:
            image = tf.image.random_flip_left_right(image)
        #pylint: disable=no-member
        image.set_shape((image_size, image_size, 3))
        image = tf.image.per_image_standardization(image)
        images_and_labels.append([image, label])

    image_batch, label_batch = tf.train.batch_join(
        images_and_labels, batch_size=batch_size,
        capacity=4 * nrof_preprocess_threads * batch_size,
        allow_smaller_final_batch=True)
  
    return image_batch, label_batch
  
def _add_loss_summaries(total_loss):
    """Add summaries for losses.
  
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
  
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
  
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
  
    return loss_averages_op


def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                # else:
                #     return learning_rate

        return learning_rate

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
  
def get_dataset(paths):
    dataset = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                image_paths = [os.path.join(facedir,img) for img in images]
                dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

def get_huge_dataset(paths, start_n, end_n):
    dataset = []
    classes = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
	for (d_ino, d_off, d_reclen, d_type, d_name) in python_getdents.getdents64(path_exp):
            if d_name=='.' or d_name == '..':
                continue
            classes += [d_name]

        classes.sort()
       	nrof_classes = len(classes)
        if end_n == -1:
            end_n = nrof_classes
        if end_n>nrof_classes:
            raise ValueError('Invalid end_n:%d more than nrof_class:%d'%(end_n,nrof_classes))      
        for i in range(start_n,end_n):
            if(i%1000 == 0):
                print('reading identities: %d/%d\n'%(i,end_n))
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                image_paths = [os.path.join(facedir,img) for img in images]
                dataset.append(ImageClass(class_name, image_paths))


    return dataset



def split_dataset(dataset, split_ratio, mode):
    if mode=='SPLIT_CLASSES':
        nrof_classes = len(dataset)
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)
        split = int(round(nrof_classes*split_ratio))
        train_set = [dataset[i] for i in class_indices[0:split]]
        test_set = [dataset[i] for i in class_indices[split:-1]]
    elif mode=='SPLIT_IMAGES':
        train_set = []
        test_set = []
        min_nrof_images = 2
        for cls in dataset:
            paths = cls.image_paths
            np.random.shuffle(paths)
            split = int(round(len(paths)*split_ratio))
            if split<min_nrof_images:
                continue  # Not enough images for test set. Skip class...
            train_set.append(ImageClass(cls.name, paths[0:split]))
            test_set.append(ImageClass(cls.name, paths[split:-1]))
    else:
        raise ValueError('Invalid train/test split mode "%s"' % mode)
    return train_set, test_set

def load_model(model_dir, meta_file, ckpt_file):
    model_dir_exp = os.path.expanduser(model_dir)
    saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
    saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))
    
# def get_model_filenames(model_dir):
#     files = os.listdir(model_dir)
#     meta_files = [s for s in files if s.endswith('.meta')]
#     if len(meta_files)==0:
#         raise ValueError('No meta file found in the model directory (%s)' % model_dir)
#     elif len(meta_files)>1:
#         raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
#     meta_file = meta_files[0]
#     ckpt_file = tf.train.get_checkpoint_state(model_dir).model_checkpoint_path
#     return meta_file, ckpt_file



def store_revision_info(src_path, output_dir, arg_string):
  
    # #  git hash
    # gitproc = Popen(['git', 'rev-parse', 'HEAD'], stdout = PIPE, cwd=src_path)
    # (stdout, _) = gitproc.communicate()
    # git_hash = stdout.strip()
    #
    # # Get local changes
    # gitproc = Popen(['git', 'diff', 'HEAD'], stdout = PIPE, cwd=src_path)
    # (stdout, _) = gitproc.communicate()
    # git_diff = stdout.strip()
    
    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        # text_file.write('git hash: %s\n--------------------\n' % git_hash)
        # text_file.write('%s' % git_diff)

def list_variables(filename):
    reader = training.NewCheckpointReader(filename)
    variable_map = reader.get_variable_to_shape_map()
    names = sorted(variable_map.keys())
    return names

## get the labels of  the triplet paths for calculating the center loss - mzh edit 31012017
def get_label_triplet(triplet_paths):
    classes = []
    classes_list = []
    labels_triplet = []
    for image_path in triplet_paths:
        str_items=image_path.split('/')
        classes_list.append(str_items[-2])

    classes = list(sorted(set(classes_list), key=classes_list.index))

    for item in classes_list:
        labels_triplet.append(classes.index(item))

    return  labels_triplet

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def class_filter(image_list, label_list, num_imgs_class):
    counter = Counter(label_list)
    label_num = counter.values()
    label_key = counter.keys()

    idx = [idx for idx, val  in enumerate(label_num)  if val > num_imgs_class]
    label_idx = [label_key[i] for i in idx]
    idx_list = [i for i in range(0,len(label_list)) if label_list[i] in label_idx]
    label_list_new = [label_list[i] for i in idx_list]
    image_list_new = [image_list[i] for i in idx_list]

    #plt.hist(label_num, bins = 'auto')
    return image_list_new, label_list_new


