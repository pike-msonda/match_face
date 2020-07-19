# from easyfacenet.simple import facenet
import face_recognition as fcr
from scipy import spatial
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
def getFaces():
    return [
        # 'data/face.jpg',
        'data/face.jpg',
        'data/face1.jpg'
    ]
def compare_face(images, threshold=0.7):
    emb = facenet.embedding(images)
    sims = np.zeros((len(images), len(images)))
    scores = np.zeros((len(images), len(images)))

    for i in range(len(images)):
        for j in range(len(images)):
            sims[i][j] = (1 - spatial.distance.cosine(emb[i], emb[j]) > threshold)
            scores[i][j] = spatial.distance.cosine(emb[i], emb[j])
            

    return sims, scores

def readImages(paths):
    images = []
    for img in paths:
        images.append(cv2.resize(cv2.imread(img), (160, 160)))
    return images

if __name__ == "__main__":
    print(getFaces())
    # aligned = facenet.align_face(getFaces())
    # emb = facenet.embedding(aligned)
    # originalImages = readImages(getFaces())
    # id_image = fcr.load_image_file(getFaces()[0])
    # selfie_image = fcr.load_image_file(getFaces()[1])
    faces = []
    for img_path in getFaces():
        image = fcr.load_image_file(img_path)
        face_location = fcr.face_locations(image)
        if len(face_location) <  1:
            raise Exception("Image has no face, please upload proper image")

        top, right, bottom, left = fcr.face_locations(image)[0]
        face = image[top:bottom, left:right]
        faces.append(face)
    # comparisons, scores = compare_face(aligned)
    # print(scores)
    # print(comparisons)
    # fig, axes = plt.subplots(2, 2)
    # axes[0, 0].imshow(aligned[0])
    # axes[0, 0].set_title('Comparison score to Selfie Image:'+ str(scores[0][1]))

    # axes[0, 1].imshow(aligned[1])
    # axes[0, 1].set_title('Comparison score to ID image:'+ str(scores[1][0]))

    # axes[1, 0].imshow(originalImages[0])
    # axes[1, 0].set_title('Original ID Image')

    # axes[1, 1].imshow(originalImages[1])
    # axes[1, 1].set_title('Original Selfie Image')

    id_encoding = fcr.face_encodings(faces[0])[0]
    face_encoding = fcr.face_encodings(faces[0])[0]
    results = fcr.compare_faces([id_encoding], face_encoding)
    print(results)
    import pdb; pdb.set_trace()
    plt.show()