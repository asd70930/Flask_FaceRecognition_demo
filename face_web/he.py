import face_model
import numpy as np
import base64
import cv2
model = face_model.FaceModel(image_size='112,112', model='model,0')
PREBASE64 = "data:image/jpeg;base64,"


def recognition(data):
    state = 0
    ans = False
    basea = data["imgs"]["a"].split(",")[1]
    baseb = data["imgs"]["b"].split(",")[1]
    print('data get!')
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

    if dist <= 0.8:
        ans = True
    return {"state": state, "basea": out_basea, "baseb": out_baseb, "ans": ans}

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