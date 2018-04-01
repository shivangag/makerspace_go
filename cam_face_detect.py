from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import cv2
import random
from time import sleep
from tensorflow.python.platform import gfile
import pickle

def prewhiten_and_expand(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    y = np.expand_dims(y, 0)
    return y  

def main(args):
    sleep(random.random())
    cam = cv2.VideoCapture(args.cam_device)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
    print('Creating networks and loading parameters')
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
            facenet.load_model(args.facenet_model_file)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            classifier_filename_exp = os.path.expanduser(args.classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
            print('Loaded classifier model from file "%s"' % classifier_filename_exp)

            minsize = 20 # minimum size of face
            threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
            factor = 0.709 # scale factor
            file_index = 0
            while cv2.waitKey(10) != 'q':
                retval, img = cam.read()
                if(retval):
                    if img.ndim<2:
                        print('Unable to align')
                        continue
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                    img = img[:,:,0:3]

                    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    if nrof_faces>0:
                        det = bounding_boxes[:,0:4]
                        det_arr = []
                        img_size = np.asarray(img.shape)[0:2]
                        if nrof_faces>1:
                            if args.detect_multiple_faces:
                                for i in range(nrof_faces):
                                    det_arr.append(np.squeeze(det[i]))
                            else:
                                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                                img_center = img_size / 2
                                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                                index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                                det_arr.append(det[index,:])
                        else:
                            det_arr.append(np.squeeze(det))

                        for i, det in enumerate(det_arr):
                            det = np.squeeze(det)
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0]-args.margin/2, 0)
                            bb[1] = np.maximum(det[1]-args.margin/2, 0)
                            bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                            bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                            scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                            scaled = prewhiten_and_expand(scaled)
                            emb = sess.run(embeddings, feed_dict={images_placeholder:scaled, phase_train_placeholder:False})
                            predictions = model.predict_proba(emb)
                            best_class_indices = np.argmax(predictions)
                            best_class_probabilities = predictions[0, best_class_indices]
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0,255,0), 5)
                            cv2.putText(img,class_names[best_class_indices] ,(bb[0], bb[1] - 10), font, 0.5, (255,0,0),2 ,cv2.LINE_AA)
                    else:
                        print('No face detected')

                    cv2.imshow('Detection',img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            cap.release()
            cv2.destroyAllWindows()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=True)
    parser.add_argument('--cam_device', type=int,
        help='Cameradevice ID.', default=0)
    parser.add_argument('--facenet_model_file', type=str,
        help='Facenet model file path', default='/home/shivang/DevPro/FaceRecognition/20170512-11054/20170512-110547.pb')
    parser.add_argument('--classifier_filename', type=str,
        help='Facenet model file path', default='/home/shivang/DevPro/facenet/lfw_classifier.pkl')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))