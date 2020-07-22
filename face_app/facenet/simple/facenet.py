import tensorflow as tf
from .. src.align import detect_face
from .. src import facenet
from .. simple import download_model
import sys
import os
from os.path import expanduser
import copy
import cv2
import numpy as np
from scipy import spatial

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor


def align_face(images, image_size=160, margin=11):
    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    tmp_image_paths = copy.copy(images)
    img_list = []
    for image in tmp_image_paths:
        img = cv2.imread(os.path.expanduser(image))[:, :, ::-1]
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(
            img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            tmp_image_paths.remove(image)
            print("can't detect face, remove ", image)
            continue
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = cv2.resize(cropped[:, :, ::-1],
                             (image_size, image_size))[:, :, ::-1]
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images


def embedding(images):
    # check is model exists
    home = expanduser('~')
    model_path = home + '/.facenet_model/20180408-102900/20180408-102900.pb'
    if not os.path.exists(model_path):
        print("model not exists, downloading model")
        download_model.download()
        print("model downloaded to " + model_path)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(model_path)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: images,
                         phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)

    return emb


def compare(images, threshold=0.7):
    emb = embedding(images)

    sims = np.zeros((len(images), len(images)))
    for i in range(len(images)):
        for j in range(len(images)):
            sims[i][j] = (
                1 - spatial.distance.cosine(emb[i], emb[j]) > threshold)

    return sims
