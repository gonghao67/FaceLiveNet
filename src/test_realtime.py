#############   face_expression ######################################################################
#####  FER2013 :  0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral             ######
#####  CK+: 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise#  ######
#############   face_verification, face_expression ####################################################
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
from scipy import misc

import time


import align.face_align_mtcnn
import face_verification
import facenet




def verification_test(args):
    # label = []
    # phrase = []
    # with open('/mnt/hgfs/VMshare-2/fer2013/fer2013.csv', 'rb') as csvfile:
    #     reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    #     header = next(reader)
    #     for row in reader:
    #         label.append(row[0])
    #         img = row[1]
    #         img = img.split(' ')
    #         img = [int(i) for i in img]
    #         img = np.array(img)
    #         img = img.reshape(48,48)
    #         phrase.append(row[2])

    predict_issame = False
    rect_len = 120
    offset_x = 50
    offset_y = 40

    # Expr_str = ["Neu", "Ang", "Cont", "Disg", "Fear", "Hap", "Sad", "Surp"] ###CK+
    # Expr_dataset = 'CK+'
    #Expr_str = ["Ang", "Disg", "Fear", "Hap", "Sad", "Surp", "Neu"]  ###FER2013+
    #Expr_str = ['Neu', 'Ang', 'Disg', 'Fear', 'Hap', 'Sad', 'Surp'] #####FER2013+ EXPRSSIONS_TYPE_fusion
    Expr_str = ['Neutre', 'Colere', 'Degoute', 'Peur', 'Content', 'Triste', 'Surprise']  #####FER2013+ EXPRSSIONS_TYPE_fusion
    Expr_dataset = 'FER2013'

    c_red = (0, 0, 255)
    c_green = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    #express_probs = np.ones(7)

    scale_size = 3  ## scale the original image as the input image to align the face


    ## load models for the face detection and verfication

    pnet, rnet, onet, sess, args_model = face_verification.load_models_forward_v2(args, Expr_dataset)


    face_img_refs_ = []
    img_ref_paths = []
    probs_face = []
    for img_ref_path in os.listdir(args.img_ref):
        img_ref_paths.append(img_ref_path)
        img_ref = misc.imread(os.path.join(args.img_ref, img_ref_path))  # python format
        img_size = img_ref.shape[0:2]

        bb, probs = align.face_align_mtcnn.align_mtcnn_realplay(img_ref, pnet, rnet, onet)
        if (bb == []):
            continue;

        bb_face = []
        probs_face = []
        for i, prob in enumerate(probs):
            if prob > args.face_detect_threshold:
                bb_face.append(bb[i])
                probs_face.append(prob)

        bb = np.asarray(bb_face)
        probs = np.asarray(probs_face)

        det = bb

        if det.shape[0] > 1:
            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = np.array(img_size) / 2
            offsets = np.vstack(
                [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            index = np.argmax(
                bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
            det = det[index, :]
            prob = probs_face[index]

        det = np.squeeze(det)
        x0 = det[0]
        y0 = det[1]
        x1 = det[2]
        y1 = det[3]

        bb_tmp = np.zeros(4, dtype=np.int32)
        bb_tmp[0] = np.maximum(det[0] - args.margin / 2, 0)
        bb_tmp[1] = np.maximum(det[1] - args.margin / 2, 0)
        bb_tmp[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
        bb_tmp[3] = np.minimum(det[3] + args.margin / 2, img_size[0])

        face_img_ref = img_ref[bb_tmp[1]:bb_tmp[3], bb_tmp[0]:bb_tmp[2], :]
        face_img_ref = misc.imresize(face_img_ref, (args.image_size, args.image_size), interp='bilinear')
        face_img_ref_ = facenet.load_data_im(face_img_ref, False, False, args.image_size)
        face_img_refs_.append(face_img_ref_)

        img_ref_cv = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
        cv2.rectangle(img_ref_cv, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), c_red, 2, 8, 0)
        # cv2.putText(img_ref_, "%.4f" % prob, (int(x0), int(y0)), font, 1, c_green, 3)
        img_ref_name = img_ref_path.split('.')[0]
        cv2.putText(img_ref_cv, "%s" % img_ref_name, (int(x0), int(y0 - 10)), font,
                    1,
                    c_red, 2)
        cv2.imshow('%s'%img_ref_path, img_ref_cv)
        cv2.waitKey(20)

    face_img_refs_ = np.array(face_img_refs_)

    #emb_ref = face_verification.face_embeddings(face_img_refs_,args, sess, images_placeholder, embeddings, keep_probability_placeholder, phase_train_placeholder)
    emb_ref = face_verification.face_embeddings(face_img_refs_, args, sess, args_model, Expr_dataset)


    ################ capture the camera for realplay #############################################
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    realplay_window = "Realplay"
    cv2.namedWindow(realplay_window, cv2.WINDOW_NORMAL)


    while (True):
        if cv2.getWindowProperty(realplay_window, cv2.WINDOW_NORMAL) < 0:
            return
        # Capture frame-by-frame
        t6 = time.time()
        ret, frame = cap.read()
        t7 = time.time()
        #print('face cap eclapse %f, FPS:%d' % ((t7 - t6), int(1 / ((t7 - t6)))))

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame is None:
            continue;



        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #img = Image.fromarray(cv2_im)

        #face alignmentation
        #im_np = np.asarray(img)
        im_np = cv2_im
        img_size = im_np.shape[0:2]
        im_np_scale = cv2.resize(im_np, (int(img_size[1] / scale_size), int(img_size[0] / scale_size)),
                                  interpolation=cv2.INTER_LINEAR)
        t0 = time.time()
        bb, probs = align.face_align_mtcnn.align_mtcnn_realplay(im_np_scale, pnet, rnet, onet)
        t1 = time.time()
        #print('align face FPS:%d' % (int(1 / ((t1 - t0)))))

        bb_face = []
        probs_face = []
        for i, prob in enumerate(probs):
            if prob > args.face_detect_threshold:
                bb_face.append(bb[i])
                probs_face.append(prob)

        bb = np.asarray(bb_face)
        probs = np.asarray(probs_face)

        bb = bb*scale_size #re_scale of the scaled image for align_face

        if (len(bb) > 0):
            for i in range(bb.shape[0]):
                prob = probs[i]
                det = bb[i]
                bb_tmp = np.zeros(4, dtype=np.int32)
                bb_tmp[0] = np.maximum(det[0] - args.margin / 2, 0)
                bb_tmp[1] = np.maximum(det[1] - args.margin / 2, 0)
                bb_tmp[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
                bb_tmp[3] = np.minimum(det[3] + args.margin / 2, img_size[0])

                face_img = im_np[bb_tmp[1]:bb_tmp[3], bb_tmp[0]:bb_tmp[2], :]
                face_img_ = misc.imresize(face_img, (args.image_size, args.image_size), interp='bilinear')
                face_img_ = facenet.load_data_im(face_img_, False, False, args.image_size)


                #########
                x0 = bb[i][0]
                y0 = bb[i][1]
                x1 = bb[i][2]
                y1 = bb[i][3]
                offset_y = int((y1-y0)/7)


                # if (predict_issame):
                #     cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), c_red, 2,
                #                   8,
                #                   0)
                #     cv2.putText(frame, "%.4f" % prob, (int(x0), int(y0)), font,
                #                 1,
                #                 c_red, 3)
                #
                #     for k in range(express_probs.shape[0]):
                #         cv2.putText(frame, Expr_str[k], (int(x1 + offset_x / 4), int(y0 + offset_y * k + offset_y / 4)), font,
                #                     0.5,
                #                     c_red, 2)
                #         cv2.rectangle(frame, (int(x1 + offset_x), int(y0 + offset_y * k)),
                #                       (int(x1 + offset_x + rect_len * express_probs[i]), int(y0 + offset_y * k + offset_y/2)),
                #                       c_red, cv2.FILLED,
                #                       8,
                #                       0)
                #
                # else:
                #     cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), c_green, 2,
                #                   8,
                #                   0)
                #     cv2.putText(frame, "%.4f" % prob, (int(x0), int(y0)), font,
                #                 1,
                #                 c_green, 3)
                #
                #     for k in range(express_probs.shape[0]):
                #         cv2.putText(frame, Expr_str[k], (int(x1 + offset_x / 4), int(y0 + offset_y * k + offset_y / 4)),
                #                     font,
                #                     0.5,
                #                     c_green, 2)
                #         cv2.rectangle(frame, (int(x1 + offset_x), int(y0 + offset_y * k)),
                #                       (int(x1 + offset_x + rect_len * express_probs[i]),
                #                        int(y0 + offset_y * k + offset_y / 2)),
                #                       c_green, cv2.FILLED,
                #                       8,
                #                       0)

                # face experssion
                ##### 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise   ############
                t2 = time.time()
                #predict_issames, dists, express_probs = face_verification.face_expression_multiref_forward(face_img_, emb_ref, args, sess, images_placeholder, embeddings, keep_probability_placeholder, phase_train_placeholder, logits)
                predict_issames, dists, express_probs = face_verification.face_expression_multiref_forward(face_img_, emb_ref, args, sess, args_model, Expr_dataset)
                t3 = time.time()


                print('face verif FPS:%d' % (int(1 / ((t3 - t2)))))

                predict_issame_idx = [i for i, predict_issame in enumerate(predict_issames) if predict_issame == True]

                if predict_issame_idx:
                    for i in predict_issame_idx:
                        dist = dists[i]
                        img_ref_name = img_ref_paths[i].split('.')[0]

                        cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), c_green, 2,
                                      8,
                                      0)
                        cv2.putText(frame, "%.4f" % prob, (int(x0), int(y0)), font,
                                    0.5,
                                    c_green, 1)
                        cv2.putText(frame, "%.2f" % dist, (int(x1), int(y1)), font,
                                    0.5,
                                    c_green, 1)
                        cv2.putText(frame, "%s" % img_ref_name, (int((x1 + x0) / 2), int(y0 - 10)), font,
                                    1,
                                    c_green, 2)

                        for k in range(express_probs.shape[0]):
                            cv2.putText(frame, Expr_str[k],
                                        (int(x1 + offset_x / 4), int(y0 + offset_y * k + offset_y / 4)),
                                        font,
                                        0.5,
                                        c_green, 1)
                            cv2.rectangle(frame, (
                            int(x1 + offset_x / 4), int(y0 + offset_y * k + offset_y / 4 + offset_y / 5)),
                                          (int(x1 + offset_x / 4 + rect_len * express_probs[k]),
                                           int(y0 + offset_y * k + + offset_y / 4 + offset_y / 2)),
                                          c_green, cv2.FILLED,
                                          8,
                                          0)
                else:
                    dist = min(dists)
                    cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), c_red, 2,
                                  8,
                                  0)
                    cv2.putText(frame, "%.4f" % prob, (int(x0), int(y0)), font,
                                0.5,
                                c_red, 1)
                    cv2.putText(frame, "%.2f" % dist, (int(x1), int(y1)), font,
                                0.5,
                                c_red, 1)

                    for k in range(express_probs.shape[0]):
                        cv2.putText(frame, Expr_str[k], (int(x1 + offset_x / 4), int(y0 + offset_y * k + offset_y / 4)),
                                    font,
                                    0.5,
                                    c_red, 1)
                        cv2.rectangle(frame, (int(x1 + offset_x / 4), int(y0 + offset_y * k + offset_y / 4 + offset_y / 5)),
                                      (int(x1 + offset_x / 4 + rect_len * express_probs[k]),
                                       int(y0 + offset_y * k + + offset_y / 4 + offset_y / 2)),
                                      c_red, cv2.FILLED,
                                      8,
                                      0)

                # for i in range(len(predict_issames)):
                #
                #     predict_issame = predict_issames[i]
                #     dist= dists[i]
                #     img_ref_name = img_ref_paths[i].split('.')[0]
                #
                #
                #
                #     if (predict_issame):
                #         cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), c_green, 2,
                #                       8,
                #                       0)
                #         cv2.putText(frame, "%.4f" % prob, (int(x0), int(y0)), font,
                #                     0.5,
                #                     c_green, 1)
                #         cv2.putText(frame, "%.2f" % dist, (int(x1), int(y1)), font,
                #                     0.5,
                #                     c_green, 1)
                #         cv2.putText(frame, "%s" % img_ref_name, (int((x1+x0)/2), int(y0-10)), font,
                #                     1,
                #                     c_green, 2)
                #
                #
                #         for k in range(express_probs.shape[0]):
                #             cv2.putText(frame, Expr_str[k], (int(x1 + offset_x / 4), int(y0 + offset_y * k + offset_y / 4)),
                #                         font,
                #                         0.5,
                #                         c_green, 1)
                #             cv2.rectangle(frame, (int(x1 + offset_x/4), int(y0 + (offset_y+offset_y/2)* k)),
                #                           (int(x1 + offset_x/4 + rect_len * express_probs[k]),
                #                            int(y0 + (offset_y+offset_y/2) * k + offset_y / 3)),
                #                           c_green, cv2.FILLED,
                #                           8,
                #                           0)
                #         break
                #
                #     else:
                #         cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), c_red, 2,
                #                       8,
                #                       0)
                #         cv2.putText(frame, "%.4f" % prob, (int(x0), int(y0)), font,
                #                     0.5,
                #                     c_red, 1)
                #         cv2.putText(frame, "%.2f" % dist, (int(x1), int(y1)), font,
                #                     0.5,
                #                     c_red, 1)
                #
                #     for k in range(express_probs.shape[0]):
                #         cv2.putText(frame, Expr_str[k], (int(x1 + offset_x / 4), int(y0 + offset_y * k + offset_y / 4)),
                #                     font,
                #                     0.5,
                #                     c_red, 1)
                #         cv2.rectangle(frame, (int(x1 + offset_x/4), int(y0 + offset_y * k + offset_y / 4 + offset_y / 5)),
                #                       (int(x1 + offset_x/4 + rect_len * express_probs[k]),
                #                        int(y0 + offset_y * k + + offset_y / 4 + offset_y / 2)),
                #                       c_red, cv2.FILLED,
                #                       8,
                #                       0)



        # visualation
        cv2.imshow(realplay_window, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    ################ capture the camera for realplay #############################################




    return



def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines():
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

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


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_ref', type=str, help='Directory with unaligned image 1.', default='../data/images')
    #parser.add_argument('--img2', type=str, help='Directory with unaligned image 2.')
    # parser.add_argument('--gpu_memory_fraction', type=float,
    #                     help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.0001)

    ## face_align_mtcnn_test() arguments
    parser.add_argument('--align_model_dir', type=str,
                        help='Directory containing the models for the face detection', default='../../model')
    parser.add_argument('--model_dir', type=str,
                        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters', default='../../model/20180115-025629_model/best_model') #20170920-174400_expression') #../model/20170501-153641#20161217-135827#20170131-234652
    parser.add_argument('--threshold', type=float,
                        help='The threshold for the face verification',default=0.9)
    parser.add_argument('--face_detect_threshold', type=float,
                        help='The threshold for the face detection', default=0.9)
    # parser.add_argument('--model_def', type=str,
    #                     help='Model definition. Points to a module containing the definition of the inference graph.',
    #                     default='models.inception_resnet_v1')
    # parser.add_argument('--weight_decay', type=float, help='L2 weight regularization.', default=5e-5)

    ## face_align_mtcnn_test() arguments
    parser.add_argument('--output_dir', type=str, help='Directory with aligned face thumbnails.', default='./align/output')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=32)
    parser.add_argument('--random_order',
                        help='Shuffles the order of images to enable alignment using multiple processes.',
                        action='store_true')


    return parser.parse_args(argv)

if __name__ == '__main__':
    verification_test(parse_arguments(sys.argv[1:]))