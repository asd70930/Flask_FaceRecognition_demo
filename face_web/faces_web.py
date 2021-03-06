#!/usr/bin/python
# -*- coding: utf-8 -*-
import base64
from flask import Flask ,render_template ,request
from flask_bootstrap import Bootstrap
from flask_moment import Moment
import cv2
import numpy as np
from os import listdir
import he
import time

UPLOAD_FOLDER = 'static/face_test'
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
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif']



##############################
bootstrap = Bootstrap(app)
moment = Moment(app)

@app.route('/')
def hello_world():
    imga = cv2.imread('static/A.png')
    imgb = cv2.imread('static/B.png')
    basea = cv2_strbase64(imga)
    baseb = cv2_strbase64(imgb)

    return render_template('my_test.html', basea=basea, baseb=baseb)

@app.route('/white_black')
def black_whitelist():
    imga = cv2.imread('static/A.png')
    imgb = cv2.imread('static/B.png')
    basea = cv2_strbase64(imga)
    baseb = cv2_strbase64(imgb)
    return render_template('white_blacklist.html', baseb=baseb, basea=basea)


@app.route('/face_recognition', methods=['POST'])
def Recognition_face():
    if request.method == "POST":
        # print('wow i get it')
        data = request.get_json()
        ans = he.recognition(data)
        return ans
    else:
        return "Error, pls using post"

@app.route('/upload_img', methods=['POST'])
def upload_imgfile():
    if request.method == "POST":
        data = request.get_json()
        ans = he.locate_face(data, UPLOAD_FOLDER)
        return ans
    else:
        return "Error, pls using post"

@app.route('/refresh_select', methods=['GET'])
def refresh_select():
    files = listdir(UPLOAD_FOLDER)
    output = {}
    file_list = []
    for file in files:
        try:
            file_type = file.split('.')[-1]
            if file_type == "jpg":
                file_list.append(file)
        except:
            continue
    output["files"] = file_list
    return output

@app.route('/show_img', methods=['POST'])
def show_img():
    if request.method == "POST":
        data = request.get_json()
        filename = data["filename"]
        image = cv2.imread(UPLOAD_FOLDER+'/'+filename)
        output_base = cv2_strbase64(image)
        return {"base": output_base}
    return {"base": "error"}

@app.route('/recognition_allface', methods=['POST'])
def recognition_allface():
    if request.method == "POST":
        data = request.get_json()
        ans  = he.recognition_all(data)
        return ans
    return {"base": "error"}

@app.route('/show_ipc', methods=['POST'])
def showIpc():
    if request.method == "POST":
        data = request.get_json()
        ans  = get_ipc(data)
        return ans
    return {"base": "error"}

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
    base64_str = cv2.imencode('.jpg', image)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    return base64_str

def cv2_strbase64(image):
    base64_str = cv2.imencode('.jpg', image)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    output = PREBASE64+str(base64_str).split("'")[1]
    return output


def get_ipc(data):
    start = time.time()
    ip = data["ip"]
    cap = cv2.VideoCapture(ip)
    ret, image = cap.read()
    if ret:
        out_base = cv2_strbase64(image)
        end = time.time()
        # print("backend spend %f" % (end - start))
        return {"ret": ret, "base": out_base}
    end = time.time()
    # print("backend spend %f" % (end - start))
    return {"ret": ret}
##################################################################

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', processes=True, threaded=False, port=5000, debug=False)


