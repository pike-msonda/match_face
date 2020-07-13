from easyfacenet.simple import facenet
import face_recognition as fcr
from scipy import spatial
import matplotlib.pyplot as plt
import cv2
import numpy as np

class FaceCompare:

    def __init__(self, id_image, selfie_image,threshold=0.7):
        self.id_image = id_image
        self.selfie_image = selfie_image
        self.threshold = threshold

    def compare(self):
        results = {}
        images = [self.id_image, self.selfie_image]
        aligned = facenet.align_face(images)
        comparisons, scores = self.distance(aligned)
        results = {
            'data':
            {
                'score': 1 - scores[0][1],
                'threshold': self.threshold,
                'verdict': bool(comparisons[0][1]),
            }
        }
        return results

    def distance(self, images):
        emb = facenet.embedding(images)
        sims = np.zeros((len(images), len(images)))
        scores = np.zeros((len(images), len(images)))

        for i in range(len(images)):
            for j in range(len(images)):
                sims[i][j] = (1 - spatial.distance.cosine(emb[i], emb[j]) > self.threshold)
                scores[i][j] = spatial.distance.cosine(emb[i], emb[j])
        return sims, scores

    def readImages(self, paths):
        images = []
        for img in paths:
            images.append(cv2.resize(cv2.imread(img), (160, 160)))
        return images

    