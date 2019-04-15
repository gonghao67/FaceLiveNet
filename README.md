# Demo of the FaceLiveNet1.0 
This demo shows the function of the FaceLiveNet for the face authentication, which employs the face verification and the liveness control simultaneous. The demo is realized in TensorFlow1.6, Python 2.7 and openCV 3.0 under Unbuntu 16.4. The details of the FaceLiveNet are described in the paper
["FaceLiveNet: End-to-End Face Verification Networks Combining With Interactive Facial Expression-based Liveness Detection"](https://www.researchgate.net/publication/325229686_FaceLiveNet_End-to-End_Face_Verification_Networks_Combining_With_Interactive_Facial_Expression-based_Liveness_Detection). The FaceLiveNet is the holistic end-to-end deep networks for the face authentication including the face verification and the liveness-control of the real presentation of the user. The Challenge-Response mechanism based on the facial expression recognition (Happy and Surprise) is used for the liveness control.

The accuracy of the face verification are measured on the benchmark LFW (99.4%) and YTF(95%), the accuracy of the facial expression recognition of the six basic expressions is measured on the benchmark CK+(99.1%), OuluCasia(87.5%), SFEW(53.2%) and FER2013(68.6%). Fusing the face verification and the facial expression recognition, the global accuracy of the face authentication on the proposed dataset based on the CK+ and the OuluCasia is 99% and 92% respectively. The proposed architecture is shown in ![Fig.1](https://github.com/zuhengming/face_recognition.git/master/figs/fig1.png). More details can be found in the paper. 

## Dependencies
- The code is tested on Ubuntu 16.04.

- install Tensorflow 1.6 (with CPU)

- install opencv 2.4.13.

- install python 2.7

## Protocol for the face authentication
The face authentication including the face verification and the liveness control can employ in two modes: real-time mode and off-line mode. For the liveness control, both two modes are based on the facial expression Challenge-Response mechanism, i.e. the system randomly proposes an expression as a request, if the user can give a right response by acting the right expression and verified by the system, the user can pass the liveness control. Specially, in this demo maximum two expressions (Happy/Surprise) can be used as the request in the liveness control. Beside the required expression, the neutral expression is always detected in both of the two modes. Since people normally start from the neutral expression to act a facial expression. In this way, the system can protect from the attack of the photo with a facial expression.

### 1. Real-time face authentication 
In the real-time face authentication mode, the face authentication will employ on the real-time camera video stream, the system will not unlock until the user gives the right response and verified by the system. 

### 2. Off-line face authentication based on the upload video
In the off-line face authentication mode, the face authentication is based on the use's upload video-clips. One expression is corresponding to one video. The user will take a video-clip of her/his facial expression and upload to the backend for the face authentication.  This mode is risk to take an inappropriate or incomplete video-clip, however the system only processes a small video-clip rather than the video stream in the real-time mode.
 
## Pretrained model
The pretrained model for FaceLiveNet1.0 is [here](https://drive.google.com/file/d/1B-ZRtWk1UoAQXHTewhKV5UPvwP3L102X/view?usp=sharing)


## Training
The face verification networks is trained on the [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html), [MSCeleb](https://www.msceleb.org/), the facial expression recognition networks branch is trained on the  [CK+](http://www.consortium.ri.cmu.edu/ckagree/), [OuluCasia](http://www.cse.oulu.fi/CMV/Downloads/Oulu-CASIA), [SFEW](https://computervisiononline.com/dataset/1105138659) and [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). 



## Face alignment
The face detection is implemented by the [Multi-task CNN (Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks).The paper for MTCNN)](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html).


## Parameters and Example
### Parameters:

--model_dir: Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters
--image_ref: The reference image for the face verification
--num_expression: The number of the required expressions for face authentication. The maximum num_request_expression is 2 which are the Happy and Surprise, otherwise the Happy will be chosen for face authentication	

### Examples for command line:

1. Real-time mode:
python Demo_FaceLiveNet1.0_Realtime.py --model_dir /mnt/hgfs/VMshare-2/Fer2013/20180115-025629_model/best_model/ --image_ref ../data/images/Zuheng.jpg --num_expression 1

2. Off-line video mode:
python Demo_FaceLiveNet1.0_video.py --model_dir /mnt/hgfs/VMshare-2/Fer2013/20180115-025629_model/best_model/ --image_ref ../data/images/Zuheng.jpg --num_expression 2

