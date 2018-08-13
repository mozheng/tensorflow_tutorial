import cv2
import numpy as np
import os
import random
import string


raw_dirpath = 'imageoutput/raw/'
defect_dirpath = 'imageoutput/defect/'

if not os.path.exists(raw_dirpath):
    os.makedirs(raw_dirpath)

if not os.path.exists(defect_dirpath):
    os.makedirs(defect_dirpath)

sum_pics = 100
for i in range(sum_pics):

    raw_img = np.zeros((512, 512, 1), np.uint8)
    top_x = random.randint(1, 511)
    top_y = random.randint(1, 511)
    down_x = random.randint(1, 511)
    down_y = random.randint(1, 511)
    w = random.randint(1, 6)
    cv2.rectangle(raw_img, (top_x, top_y), (down_x, down_y), (255, 255, 255), w)

    c_x = random.randint(1, 511)
    c_y = random.randint(1, 511)
    axe_x = random.randint(1, 511)
    axe_y = random.randint(1, 511)
    angle = random.randint(0, 359)
    startAngle = random.randint(0, 359)
    endAngle = random.randint(0, 359)
    cv2.ellipse(raw_img, (c_x, c_y), (axe_x, axe_y), angle, startAngle, endAngle, 255, thickness=w)

    ptx = np.random.randint(1, 511, [2])
    random_char = random.choice(string.ascii_letters)
    cv2.putText(raw_img, random_char, (1, ptx[0]), cv2.FONT_HERSHEY_SIMPLEX, 25, (255, 255, 255), thickness=w)
    random_char = random.choice(string.ascii_letters)
    cv2.putText(raw_img, random_char, (1, ptx[1]), cv2.FONT_HERSHEY_SIMPLEX, 25, (255, 255, 255), thickness=w)
    cv2.imwrite(os.path.join(raw_dirpath, str(i)+".jpg"), raw_img)

    pts = np.random.randint(1, 511, [10, 2])
    pts = pts.reshape((-1, 1, 2))
    w = random.randint(6, 10)
    cv2.polylines(raw_img, [pts], True, (0, 0, 0), thickness=w)

    cv2.imwrite(os.path.join(defect_dirpath, str(i) + ".jpg"), raw_img)
