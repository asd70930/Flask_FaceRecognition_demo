x = [1,2,3,4]
a,b,c,d = x
print(a)




# import face_model
# import cv2
# import numpy as np
#
# model = face_model.FaceModel(image_size='112,112', model='model,0')
#
# image_path = 'static/gc2.png'
#
# img = cv2.imread(image_path)
# img2 = cv2.imread("static/gc1.jpg")
#
#
# imga,imgb,loc = model.get_input_loc(img)
# imga2,imgb2,loc2 = model.get_input_loc(img2)
#
# f1 = model.get_feature(imga)
# f2 = model.get_feature(imga2)
#
# dist = np.sum(np.square(f1-f2))
# print(dist)
# sim = np.dot(f1, f2.T)
# print(sim)
#
#
# # cv2.imshow('a',loc)
# # cv2.imshow('b',img)
# # cv2.waitKey()
# # cv2.destroyAllWindows()
#
#
# print()
