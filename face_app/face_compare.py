from facenet.simple.facenet import align_face, embedding
from scipy import spatial
import numpy as np

class FaceCompare:

    def __init__(self, id_image, selfie_image,threshold=0.7):
        self.id_image = id_image
        self.selfie_image = selfie_image
        self.threshold = threshold

    def compare(self):
        results = {}
        images = [self.id_image, self.selfie_image]
        aligned = align_face(images)
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
        emb = embedding(images)
        sims = np.zeros((len(images), len(images)))
        scores = np.zeros((len(images), len(images)))

        for i in range(len(images)):
            for j in range(len(images)):
                sims[i][j] = (1 - spatial.distance.cosine(emb[i], emb[j]) > self.threshold)
                scores[i][j] = spatial.distance.cosine(emb[i], emb[j])
        return sims, scores

    