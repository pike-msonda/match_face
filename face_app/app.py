
import os

from flask import Flask, request
from face_compare import FaceCompare
from werkzeug.utils import secure_filename
from flask import jsonify
import uuid
import json
from http import HTTPStatus

app = Flask(__name__, instance_relative_config=True)

app.config.from_mapping(
    SECRET_KEY='dev',
    DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
)
app.debug = True
# a simple page that says hello
UPLOAD_FOLDER = 'data/'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


from werkzeug.exceptions import HTTPException

@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response

@app.route('/')
def hello():
    return jsonify ({
        "code": HTTPStatus.OK,
        "message": "POC for face verification"
    })
def remove_images(images):
    for img in images:
        os.remove(img)
    return images
def params_in(key, params):
    if key not in params:
        return key
    return ''
     
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def save_file(file):
     if file and allowed_file(file.filename):
         filename = str(uuid.uuid4()) + '.' + secure_filename(file.filename).split('.')[-1]
         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
         return os.path.join(app.config['UPLOAD_FOLDER'], filename)
         
@app.route('/api/match', methods=['GET' ,'POST'])
def compare():
    if request.method == 'GET':
        return jsonify ({
            "code" : HTTPStatus.OK,
            "message": "Use post for this route."
        })
    if request.method == 'POST':
        message = []
        if 'id_image'  or 'selfie_image' not in request.files:
            if 'id_image' not in request.files:
                message.append('id_image is required')
            if 'selfie_image' not in request.files:
               message.append('selfie_image is required')
            return jsonify({
                "errors": message,
                "code": HTTPStatus.UNPROCESSABLE_ENTITY
            })
        
        threshold = request.form.get('threshold', 0.7, type=float)
        id_image_path = save_file (request.files['id_image'])
        selfie_image_path = save_file (request.files['selfie_image'])
        face_matcher = FaceCompare(id_image_path, selfie_image_path, threshold)
        results = face_matcher.compare()
        if results:
            remove_images([id_image_path, selfie_image_path])
        return results
