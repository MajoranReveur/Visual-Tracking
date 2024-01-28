import numpy as np
from Detector import detect
from KalmanFilter import KalmanFilter
import cv2

kalmanFilter = KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_sdt_meas=0.1, y_sdt_meas=0.1)

capture = cv2.VideoCapture('randomball.avi')
count = 0
path = []
while capture.isOpened():
    ret, frame = capture.read()
    if ret:
        count += 1
        centers = detect(frame, count)
        predict_c = kalmanFilter.predict()
        predict_c = predict_c.astype(int)
        cv2.rectangle(frame,(predict_c[0][0] - 50,predict_c[1][0] - 50),(predict_c[0][0] + 50,predict_c[1][0] + 50),(0,0,255),5)
        for c in centers:
            update_c = kalmanFilter.update(c)
            update_c = update_c.astype(int)
            cv2.rectangle(frame,(update_c[0][0] - 50,update_c[1][0] - 50),(update_c[0][0] + 50,update_c[1][0] + 50),(255,0,0),5)
            c = c.astype(int)
            path.append((c[0][0], c[1][0]))
            cv2.circle(frame, (c[0][0],c[1][0]), 60, (0, 255, 0), 10)
        for i in range(len(path) - 1):
            cv2.line(frame, path[i], path[i + 1], (0, 0, 0), 4)
        cv2.imshow('result', frame)
        cv2.waitKey(1)
    else:
        capture.release()
        break