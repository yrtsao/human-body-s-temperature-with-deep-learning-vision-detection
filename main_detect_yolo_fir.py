import sys
# sys.path.append('D:\python\yolov5_notebook\yolov5-master\yolov5-master')
sys.path.append('E:\picture_fiting\yolov5-master\yolov5-master')



from pathlib import Path
import numpy as np
import torch
from numpy import random
from models.yolo import Model
from utils.general import set_logging, xyxy2xywh, scale_coords
from utils.google_utils import attempt_download
from utils.plots import plot_one_box_fir
import cv2
import time
import hubconf_video as hub
from PIL import Image
# import temperature as tp


# from .
class detect_yolo:
    def __init__(self):
        # self.model = hub.create(name='yolov5s', pretrained=True, channels=3, classes=80) 
        # self.model = hub.custom(path_or_model=r'D:\python\yolov5_notebook\yolov5-master\yolov5-master\runs\train\exp\weights\best.pt') 
        self.model = hub.custom(path_or_model=r'E:\picture_fiting\yolov5-master\yolov5-master\yolov5s.pt') 
        # self.model = hub.custom(path_or_model=r'\python\yolov5_notebook\yolov5-master\yolov5-master\yolov5s.pt') 
        self.model = self.model.autoshape() 
        self.model.conf = 0.5  # confidence threshold (0-1)
        self.model.iou = 0.45  # NMS IoU threshold (0-1)
    def detect_yolo_big (self, img):
        results = self.model(img)
        
        # pos = results.xyxy[0].cpu().numpy()
        pos = results.xyxy[0].cpu().numpy().tolist()
        print(pos)
        if pos :
            #results.xyxy[0] = x1,y1,x2,y2, score, 類別 
            # .cup() 當用gpu在跑時要回傳成cpu 
            # numpy()  tensor ->array
            # tolist() list -> array
            area_list = []
            for i in pos:
                area = (i[2]-i[0])*(i[3]-i[1])  
                area_list.append(area)
            # for i in pos:
            #     area = (i[2]-i[0])*(i[3]-i[1])  
            #     if i[4]>0.7 :
            #         area_list.append(area)
            #     else :        
            pos_big_index = area_list.index(max(area_list))
            pos_big = list(map(int,pos[pos_big_index]))
            # pos_big = pos[pos_big_index]
            #找出最大面積
            return pos_big,len(pos)
            
        else:
                s =[0,0,0,0,0]
                return None,0
    def detect_yolo (self, img):
        results = self.model(img)
        pos1 = results.xyxy[0][:,:4].round()
        pos2 = results.xyxy[0][:,4:6]
        pos = torch.cat([pos1,pos2],dim=1)

        # print("cat",torch.cat([results.xyxy[0][:,:4],results.xyxy[0][:,4:6]],dim=1) )
        # pos = results.xyxy[0].cpu().numpy().round()
        # pos = results.xyxy[0].cpu().numpy().tolist()
        # print(pos)
        #results.xyxy[0] = x1,y1,x2,y2, score, 類別 
        # .cup() 當用gpu在跑時要回傳成cpu 
        # numpy()  tensor ->array
        # tolist() list -> array
        return pos
    
def center(xyxy):
    x1,y1,x2,y2 = xyxy
    x = (x1 + x2)/2
    y = (y1 + y2)/2
    center = x , y
    return center


names =  ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
if __name__ == '__main__':
    isstop = True
    M = np.load('./transform/524-13/M.npy')
    # M = np.load('M.npy')
    cap = cv2.VideoCapture(0)
    cap_r = cv2.VideoCapture(1)
    t1 = time.time()
    # cap.set(cv2. CAP_PROP_FRAME_WIDTH, 2560)
    # cap.set(cv2. CAP_PROP_FRAME_HEIGHT, 1980)
    cap.set(cv2. CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2. CAP_PROP_FRAME_HEIGHT, 720)

    cap_r.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap_r.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    model  = detect_yolo()
    # print(22)
    # time.sleep(1)
    # imgs = cv2.imread(r'D:\python\yolov5_notebook\yolov5-master\yolov5-master\data\images/bus.jpg')
    # results = model.detect_yolo(imgs)
    # # gn = torch.tensor(imgs.shape)[[1, 0, 1, 0]]
    # if len(results):
    #     print('sss1',results)
    #     # print('len',len(results))
    #     # print('results[-1]',results[:, -1])
    #     # print('results[-1]',results[:, -1].unique())
    #     for c in results[:, -1].unique():
    #                 n = (results[:, -1] == c).sum()  # detections per class
    #                 s += f'{n} {names[int(c)]}s, '  # add to string
    #     for *xyxy, conf, cls in reversed(results):   
    #         line = (cls, *xyxy, conf) 
    #         print("line",line)
    #         print("xyxy:",xyxy)
    #         label = f'{names[int(cls)]} {conf:.2f}'
    #         plot_one_box(xyxy, imgs, label=label, color=colors[int(cls)], line_thickness=3)
    
    #     cv2.imshow("show", imgs)
    #     if cv2.waitKey(1) == ord('q'):  # q to quit
    #         raise StopIteration    
    # T = 0

    while(isstop):
        # 從攝影機擷取一張影像
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        ret_r, frame_r = cap_r.read()
        results = model.detect_yolo(frame)
        s = ''
        t2 =time.time()
        t_value = (t2 - t1)/10
        T = int(t_value % 7)
        if len(results):
            # print('sss1',results)
            # print('len',len(results))
            # print('results[-1]',results[:, -1])
            # print('results[-1]',results[:, -1].unique())
            for c in results[:, -1].unique():
                n = (results[:, -1] == c).sum()  # detections per class
                s += f'{n} {names[int(c)]}s, '  # add to string
            for *xyxy, conf, cls in reversed(results):   
                # line = (cls, *xyxy, conf) 
                # print("line",line)
                # print("xyxy:",xyxy)
                center_ai = center(xyxy)
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box_fir(xyxy, frame, frame_r, M, T,label=label, color=colors[int(cls)], line_thickness=3)
            cv2.putText(frame, s, (0,20), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 2)
            cv2.imshow("show", frame)
            if cv2.waitKey(1) == ord('q'): 
                isstop = False
                break  
    
    # 釋放攝影機
    cap.release()
    cap_r.release()
    # 關閉所有 OpenCV 視窗
    cv2.destroyAllWindows()
