#!/usr/bin/python
# -*- coding: utf-8 -*-
import base64
from flask import Flask ,render_template ,request ,session ,jsonify ,redirect ,url_for ,flash
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from datetime import datetime
from flask_wtf import FlaskForm
from wtforms import StringField ,SubmitField
import cv2
import numpy as np


UPLOAD_FOLDER = 'static/uploads/'
PREBASE64 = "data:image/jpeg;base64,"


app = Flask(__name__)
# 配置密鑰: 防止 CSRF (Cross-site request forgery)
# import secrets ; secrets.token_urlsafe(12)
# 設置 flask的密鑰 方法一
app.config['SECRET_KEY'] = '-XQRl_yWcNaqrVsm'
app.config['TEMPLATES_AUTO_RELOAD'] = True
# 設置 flask的密鑰 方法二
# app.secret_key = app.config.get('flask' ,'123321456789')
print("secret key : ", app.secret_key)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif']



##############################
bootstrap = Bootstrap(app)
moment = Moment(app)

class Up_img(FlaskForm):
    submit = SubmitField('上傳圖片')
@app.route('/')
def hello_world():
    imga = cv2.imread('static/A.png')
    imgb = cv2.imread('static/B.png')
    basea = cv2_strbase64(imga)
    baseb = cv2_strbase64(imgb)

    return render_template('my_test.html', basea=basea, baseb=baseb)


@app.route('/face_recognition', methods=['POST'])
def Recognition_face():
    if request.method == "POST":
        # print('wow i get it')
        data = request.get_json()

        ans = he.recognition(data)

        return ans



    imga = cv2.imread('static/A.png')
    imgb = cv2.imread('static/B.png')
    basea = cv2_strbase64(imga)
    baseb = cv2_strbase64(imgb)
    return render_template('my_test.html', basea=basea, baseb=baseb)





##################################################################
def base64toimg(string):
    """
    base64 to np type image

    :param string: base64 string
    :return: numpy image
    """
    erro  = True
    lens = len(string)
    lenx = lens - (lens % 4 if lens % 4 else 0)
    result = string[:lenx]
    try:
        base64encode = bytes(result, encoding="utf8")
        base64decode = base64.b64decode(base64encode)
        img_array = np.fromstring(base64decode, np.uint8)
        img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
        if img is None:
            return None
        else:
            return img
    except Exception as e:
        return None


def cv2_base64(image):
    base64_str = cv2.imencode('.jpg',image)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    return base64_str

def cv2_strbase64(image):
    base64_str = cv2.imencode('.jpg', image)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    output = PREBASE64+str(base64_str).split("'")[1]
    return output

##################################################################

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == "__main__":
    app.run(processes=True, threaded=False, port=5000, debug=True)


