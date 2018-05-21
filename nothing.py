import os
from firebase import firebase
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_restful import Resource, Api
from werkzeug import secure_filename
from keras.preprocessing.image import load_img, img_to_array
import uuid
import base64
import tensorflow as tf
from flask import send_from_directory
from werkzeug import SharedDataMiddleware

import predict

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

# 바이트 문자열(url) 인코딩
def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


#########################################################################


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='../uploads/main.jpg')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(file_path)
            file.save(file_path)
            result = predict.predict(file_path)

            if result == 0:
                label = "안투리움"
            elif result == 1:
                label =  "Ball Moss"
            elif result == 2:
                label = '참매발톱'
            elif result == 3:
                label = '가자니아'
            elif result == 4:
                label = "장미"
            elif result == 5:
                label = "해바라기"
            elif result == 6:
                label = "wall flower"
            elif result == 7:
                label = "yellow iris"
            
            print(result)
            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('template.html', label=label, imagesource='../uploads/' + filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)

app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})



if __name__ == "__main__":
    app.debug=True
    app.run(host='0.0.0.0', port=3000)