import face_recognition as fcr
if __name__ == "__main__":
    id_image = fcr.load_image_file('data/id.jpg')
    selfie_image = fcr.load_image_file('data/index.jpeg')
    id_encoding = fcr.face_encodings(id_image)[0];
    selfie_encoding =  fcr.face_encodings(selfie_image)[0];
    results = fcr.compare_faces([id_encoding], selfie_encoding)
    print(results)
    import pdb; pdb.set_trace()