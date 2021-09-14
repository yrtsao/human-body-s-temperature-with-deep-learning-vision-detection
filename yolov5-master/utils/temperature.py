import cv2
import numpy as np
import statistics
import heapq

np.set_printoptions(threshold=np.inf)
result = []
tph = 40   
tpl = 30
rgh = 255
rgl = 0
lim = 100
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def face_dect(image):
    perimeter_list = []
    # print("face_dect")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face = face_cascade.detectMultiScale(image, 1.1, 4)        

    if face != () :
        for (x_c, y_c, w_c, h_c) in face:
            perimeter = w_c+ h_c
            perimeter_list.append(perimeter)
        max_perimeter = perimeter_list.index(max(perimeter_list))
        # print(max_perimeter)
        face = face[max_perimeter]
        face = face.tolist()
        # print(face)
        # image = image[face[0]:face[0]+face[2],face[1]:face[1]+face[3]]

    return face

def get_temperature(image):
    # print("get_temperature")
    if image.shape[0] and image.shape[1] != 0:

        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height = image.shape[0]
        width = image.shape[1]
        result = np.zeros((height, width), np.uint8)
        # result = image[int(height/2-5):int(height/2+5),int(width/2-5):int(width/2+5)]
        for i in range(height):
            for j in range(width):
                if (int(image[i, j] > lim) and int(image[i, j] < 160)): #245,254為顏色閥值
                    gray = int(image[i, j])

                else:
                    # gray = None
                    gray = 0
                result[i, j] = gray

        
        result = result.flatten()
        # result = np.unique(result).tolist()
        result = result.tolist()
        while 0 in result:
            result.remove(0)

        # print(result)
        
        if result == []:
            temperature = "Nan" 
        
        else:
            mean = np.median(result)
            temperature = round(((mean-tpl)*(tph-tpl)/(rgh-rgl)+tpl),2)+2.5
        
       

    else:
        temperature = "Nan"

    # print(temperature)

    return temperature



def Ptransform(rect_up,rect_down,M):
    rect_up = list(rect_up) #為了加1
    rect_down = list(rect_down)

    one=[1]
    pts1 = rect_up+ one
    pts2 = rect_down+ one

    # print("pts1",pts1)
    # print("pts2",pts2)
    # print("M",M)

    up  = M.dot(pts1). astype(int)
    down  = M.dot(pts2). astype(int)
    up = up[:2]
    down = down[:2]
    # print("up",up)
    # print("down",down)

    return(up,down)

