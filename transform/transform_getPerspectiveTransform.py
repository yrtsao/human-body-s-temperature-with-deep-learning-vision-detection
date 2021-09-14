
import cv2
import numpy as np
import threading
from PIL import Image
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time
# plt.switch_backend('agg') 

xy_list = []
xy_r_list = []
cap = cv2.VideoCapture(1)
cap_r = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cap_r.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap_r.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
time.sleep(1)
def take_picture():
    while(True):
        ret, frame = cap.read()
        print(frame.shape)
        ret_r, frame_r = cap_r.read()
        print(frame_r.shape)
        cv2.imshow('frame', frame)
        cv2.imshow("image_r", frame_r)
        if cv2.waitKey(1) == ord('R'):
            cv2.imwrite('output.jpg', frame)
            cv2.imwrite('output_r.jpg', frame_r)
            break
    cap.release()
    cap_r.release()
    cv2.destroyAllWindows()
    return frame, frame_r

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    
    if event == cv2.EVENT_LBUTTONDOWN:
        xy_str = "%d,%d" % (x, y)
        xy = [x,y]
        # print (xy)
        xy_list.append(xy)
        # print(len(xy_list))
        cv2.circle(frame, (x, y), 1, (255, 0, 0), thickness = -1)
        cv2.putText(frame, xy_str, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0,0,0), thickness = 1)
        cv2.imshow("image", frame)

def on_EVENT_LBUTTONDOWN_r(event, x, y, flags, param):
    
    if event == cv2.EVENT_LBUTTONDOWN:
        xy_str = "%d,%d" % (x, y)
        xy = [x,y]
        print (xy)
        xy_r_list.append(xy)
        print(len(xy_r_list))
        cv2.circle(frame_r, (x, y), 1, (255, 0, 0), thickness = -1)
        cv2.putText(frame_r, xy_str, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0,0,0), thickness = 1)
        cv2.imshow("image_r", frame_r)
    


if __name__== '__main__':
    frame, frame_r = take_picture()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", frame)
    while(True):
        cv2.waitKey(1)
        if len(xy_list) >=4:
            cv2.destroyWindow("image")
            break

    cv2.namedWindow("image_r")
    cv2.setMouseCallback("image_r", on_EVENT_LBUTTONDOWN_r)
    cv2.imshow("image_r", frame_r)
    while(True):
        cv2.waitKey(1)
        if len(xy_r_list) >=4:
            cv2.destroyWindow("image_r")
            break    
    
    print("xy_list",xy_list)
    print("xy_list_r",xy_r_list)

    xy_list = np.array(xy_list)
    xy_list = np.float32(xy_list)

    xy_r_list = np.array(xy_r_list)
    xy_r_list = np.float32(xy_r_list)

    np.save('xy_list', xy_list)
    np.save('xy_r_list', xy_r_list)
    M = cv2.getPerspectiveTransform(xy_list, xy_r_list)

    np.save('M', M)
    print("M",M)
    print("Done.")