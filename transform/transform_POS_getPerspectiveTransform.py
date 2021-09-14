import cv2
import numpy as np
from matplotlib import pyplot as plt




def Ptransform(rect_up,rect_down,M):
    rect_up = list(rect_up) #為了加1
    rect_down = list(rect_down)

    one=[1]
    pts1 = rect_up+ one
    pts2 = rect_down+ one

    print("pts1",pts1)
    print("pts2",pts2)
    print("M",M)

    up  = M.dot(pts1). astype(int)
    down  = M.dot(pts2). astype(int)
    up = up[:2]
    down = down[:2]
    print("up",up)
    print("down",down)

    return(up,down)


if __name__ == '__main__':
    file = "512-20"
    img1 = cv2.imread('./'+file+'/output.jpg')
    img2 = cv2.imread('./'+file+'/output_r.jpg')
    M = np.load('./'+file+'/M.npy')
    xy_list_range = np.load('./'+file+'/xy_list.npy')#確認邊界
    x_max = max(xy_list_range[:,0])
    x_min = min(xy_list_range[:,0])
    y_max = max(xy_list_range[:,1])
    y_min = min(xy_list_range[:,1]) 

    print("xy_list_range",xy_list_range)

    xy_r_list_range = np.load('./'+file+'/xy_r_list.npy')#確認邊界
    x_max = max(xy_list_range[:,0])
    x_min = min(xy_list_range[:,0])
    y_max = max(xy_list_range[:,1])
    y_min = min(xy_list_range[:,1]) 

    # img3 = img1.copy()
    # img4 = img2.copy()
    while(1):
        cv2.namedWindow("image", flags= cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        rect = cv2.selectROI("image", img1, False, False)

        cv2.destroyAllWindows()
        rect_up = rect[0:2]
        rect_wh = rect[2:] 
        rect_down = [i + j for i, j in zip(rect_up, rect_wh)]

        if rect_up[0] > x_min and rect_up[1] > y_min:
            if  rect_down[0] < x_max and rect_down[1] < y_max:
                up, down = Ptransform(rect_up,rect_down,M)

                rect_up_tuple = tuple(rect_up)#為了加畫圖
                rect_down_tuple = tuple(rect_down)

                up = tuple(up)#為了加畫圖
                down = tuple(down)
                cv2.rectangle(img1,  rect_up_tuple ,rect_down_tuple , (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('SELF', img1)
                cv2.waitKey(0)

                cv2.namedWindow("SELF_AFTER", flags= cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
                cv2.rectangle(img2,  up, down , (0, 255, 0), 4, cv2.LINE_AA)
                cv2.imshow('SELF_AFTER', img2)
                cv2.waitKey(0)
                if cv2.waitKey(1) == ord('q'):
                    break