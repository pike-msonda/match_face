<<<<<<< HEAD
from easyfacenet.simple import facenet
from scipy import spatial
import matplotlib.pyplot as plt
import cv2
import numpy as np
def getFaces():
    return [
        # 'data/face.jpg',
        'data/face5.jpg',
        'data/face3.jpg'
    ]
    pass
def compare_face(images, threshold=0.7):
    emb = facenet.embedding(images)
    sims = np.zeros((len(images), len(images)))
    scores = np.zeros((len(images), len(images)))

    for i in range(len(images)):
        for j in range(len(images)):
            import pdb; pdb.set_trace()
            sims[i][j] = (1 - spatial.distance.cosine(emb[i], emb[j]) > threshold)
            scores[i][j] = spatial.distance.cosine(emb[i], emb[j])
            

    return sims, scores

def readImages(paths):
    images = []
    for img in paths:
        images.append(cv2.resize(cv2.imread(img), (160, 160)))
    return images

if __name__ == "__main__":
    aligned = facenet.align_face(getFaces())
    originalImages = readImages(getFaces())
    comparisons, scores = compare_face(aligned)
    print(scores)
    print(comparisons)
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(aligned[0])
    axes[0, 0].set_title('Comparison score to Selfie Image:'+ str(scores[0][1]))

    axes[0, 1].imshow(aligned[1])
    axes[0, 1].set_title('Comparison score to ID image:'+ str(scores[1][0]))

    axes[1, 0].imshow(originalImages[0])
    axes[1, 0].set_title('Original ID Image')

    axes[1, 1].imshow(originalImages[1])
    axes[1, 1].set_title('Original Selfie Image')
    plt.show()
=======
import face_recognition as fcr
if __name__ == "__main__":
    id_image = fcr.load_image_file('data/id.jpg')
    selfie_image = fcr.load_image_file('data/index.jpeg')
    id_encoding = fcr.face_encodings(id_image)[0];
    selfie_encoding =  fcr.face_encodings(selfie_image)[0];
    results = fcr.compare_faces([id_encoding], selfie_encoding)
    print(results)
>>>>>>> 45736553d645475b9e801186f06287de537fdf94
    import pdb; pdb.set_trace()