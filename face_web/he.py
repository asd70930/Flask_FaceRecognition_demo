#!/usr/bin/python
# -*- coding: utf-8 -*-
import face_model
import numpy as np
import base64
import cv2
from os import listdir
from os.path import join
from PIL import Image, ImageDraw, ImageFont

model = face_model.FaceModel(image_size='112,112', model='model,0')
PREBASE64 = "data:image/jpeg;base64,"
UPLOAD_FOLDER = 'static/face_test'
THRESHOULD = 1.46

def recognition_all(data):
    ans = False
    basea = data["imgs"]["a"].split(",")[1]
    imga = base64toimg(basea)
    inps, origin_img, locations, count = model.get_input_locs(imga)

    if count == 0:
        out_base = cv2_strbase64(imga)
        return {"state": 0, "base": out_base}
    else:
        out_image = imga.copy()
        face_encodeings = []
        for inp in inps:
            face_encoding = model.get_feature(inp)
            face_encodeings.append(face_encoding)

        file_names = []
        encode_list = []
        files = listdir(UPLOAD_FOLDER)
        for file in files:
            try:
                file_type = file.split('.')[-1]
                if file_type == "jpg":
                    file_names.append(file.split(".")[0])

                    image = cv2.imread(join(UPLOAD_FOLDER, file))
                    inputfirst, _ = model.get_input(image)
                    encode = model.get_feature(inputfirst)
                    encode_list.append(encode)
            except:
                continue
        known_face_encodings_copy = np.array(encode_list)

        for i, face_encoding in enumerate(face_encodeings):
            ans_name = "Unknown"
            face_distances = np.square(np.linalg.norm(known_face_encodings_copy - face_encoding, axis=1))
            # face_dist_arry = np.array(face_distances)
            matches = list(face_distances <= THRESHOULD)
            best_match_index = np.argmin(face_distances)
            print("best distance :", face_distances[best_match_index])
            if matches[best_match_index]:
                ans_name = file_names[best_match_index]

            # Display the results
            bbox = locations[i]

            left, top, right, bottom = bbox
            # Draw a box around the face

            cv2.rectangle(out_image, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            # cv2.rectangle(out_image, (left, bottom - 10), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(out_image, ans_name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)



        out_base = cv2_strbase64(out_image)
        return {"state": 1, "base": out_base}


def recognition(data):
    ans = False
    basea = data["imgs"]["a"].split(",")[1]
    baseb = data["imgs"]["b"].split(",")[1]
    imga = base64toimg(basea)
    imgb = base64toimg(baseb)

    inputa, _1, loca = model.get_input_loc(imga)
    inputb, _2, locb = model.get_input_loc(imgb)
    if inputa is None:
        state = 1
        if inputb is None:
            state = 0
            out_baseb = cv2_strbase64(imgb)
            out_basea = cv2_strbase64(imga)
            return {"state": state, "basea": out_basea, "baseb": out_baseb}
        out_baseb = cv2_strbase64(locb)
        out_basea = cv2_strbase64(imga)
        return {"state": state, "basea": out_basea, "baseb": out_baseb}

    if inputb is None:
        state = 2
        out_basea = cv2_strbase64(loca)
        out_baseb = cv2_strbase64(imgb)
        return {"state": state, "basea": out_basea, "baseb": out_baseb}

    out_basea = cv2_strbase64(loca)
    out_baseb = cv2_strbase64(locb)

    state = 3
    f1 = model.get_feature(inputa)
    f2 = model.get_feature(inputb)

    dist = np.sum(np.square(f1 - f2))
    # print(dist)

    if dist <= THRESHOULD:
        ans = True
    return {"state": state, "basea": out_basea, "baseb": out_baseb, "ans": ans}

def locate_face(data, filepath):
    """

    :param data:
    :return:
    state:0 , can't find face, do not save the image
    state:1 , only one face in image, return image base64, and save image
    state:2 , face more than 1, do not save the image
    """
    basea = data["imgs"]["a"].split(",")[1]
    imga = base64toimg(basea)
    filename = data["imgs"]["filename"]
    loc, count = model.get_locate_count(imga)
    print("find face %i" % count)
    if count ==0:
        # can't find face, do not save the image
        out_base = cv2_strbase64(imga)
        return {"state": 0, "base": out_base}
    elif count == 1:
        cv2.imwrite(filepath+'/'+filename+'.jpg', imga, [cv2.IMWRITE_JPEG_QUALITY, 90])
        out_base = cv2_strbase64(loc)
        return {"state": 1, "base": out_base}
    else:
        # face more than 1, do not save the image
        out_base = cv2_strbase64(loc)
        return {"state": 2, "base": out_base}

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


