from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import random
from  PIL import Image
import time
import shutil

sys.path.append('../')
import facenet
import align.detect_face


def load_align_mtcnn(model_dir):
    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
            # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, model_dir)

    return pnet, rnet, onet

def align_mtcnn(args, pnet, rnet, onet):

    src_path,_ = os.path.split(os.path.realpath(__file__))

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    nrof_images_total = 0
    nrof_successfully_aligned = 0

    scaled_imgs = np.zeros((2, args.image_size, args.image_size, 3))

    time_rec = []
    bboxes = np.zeros((2,5))
    for image_path in [args.img1,args.img2]:
        time_rec.append(time.time())
        nrof_images_total += 1
        filename = os.path.splitext(os.path.split(image_path)[1])[0]

        if os.path.exists(image_path):
            try:
                img = misc.imread(image_path)
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(image_path, e)
                print(errorMessage)
            else:
                if img.ndim<2:
                    print('Unable to align "%s"' % image_path)
                    continue
                if img.ndim == 2:
                    img = facenet.to_rgb(img)

                img = img[:, :, 0:3]
                output_tmp_path = os.path.join(os.path.abspath(args.output_dir),filename)

                bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor, output_tmp_path)

                nrof_faces = bounding_boxes.shape[0]
                if nrof_faces>0:
                    det = bounding_boxes[:,0:4]
                    img_size = np.asarray(img.shape)[0:2]
                    if nrof_faces>1:
                        bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                        img_center = img_size / 2
                        offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                        index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                        det = det[index,:]
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0]-args.margin/2, 0)
                    bb[1] = np.maximum(det[1]-args.margin/2, 0)
                    bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                    bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])

                    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                    scaled_imgs[nrof_successfully_aligned,:,:,:] = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                    bboxes[nrof_successfully_aligned,0:4]=bb ## bounding box with the margin
                    if nrof_faces > 1:
                        bboxes[nrof_successfully_aligned, 4] = bounding_boxes[index, 4] ## the detection score/probability of the bouding box
                    else:
                        bboxes[nrof_successfully_aligned, 4] = bounding_boxes[0, 4]


                    nrof_successfully_aligned += 1
                else:
                    print('Unable to align "%s"' % image_path)
                time_rec.append(time.time())

def align_mtcnn_realplay(img, pnet, rnet, onet):

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    bb = []
    prob = []

    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces > 0:
        det = bounding_boxes[:, 0:5]
        bb = det[:, 0:4]
        prob = det[:, 4]
        img_size = np.asarray(img.shape)[0:2]
        # if nrof_faces > 1:
        #     bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
        #     img_center = img_size / 2
        #     offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
        #                          (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
        #     offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
        #     index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
        #     # if det.shape[0]>0:
        #     #     print (det.shape)
        #     #     print (index.shape)
        #     det = det[np.newaxis, index, :]
        #     bb = det[:, 0:4]
        #     prob = det[:, 4]
    # else:
    #     print('Unable to align image')


    return bb, prob
