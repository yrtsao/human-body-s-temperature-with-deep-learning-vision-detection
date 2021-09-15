# human-body-s-temperature-with-deep-learning-vision-detection
使用 Yolo v5 深度學習模型辨識彩色影像中物品與人，並利用熱像儀量測人體溫度，輸出於畫面的搜索框上，由於彩色影像與熱像儀影像的可視範圍不一致，本專案使用透視變換進行校正。

required
---------------------------------------------------------------------
      # base ----------------------------------------
      Cython
      matplotlib>=3.2.2
      numpy>=1.18.5
      opencv-python>=4.1.2
      Pillow
      PyYAML>=5.3
      scipy>=1.4.1
      tensorboard>=2.2
      torch>=1.7.0
      torchvision>=0.8.1
      tqdm>=4.41.0

      # logging -------------------------------------
      # wandb

      # plotting ------------------------------------
      seaborn>=0.11.0
      pandas

      # export --------------------------------------
      # coremltools==4.0
      # onnx>=1.8.0
      # scikit-learn==0.19.2  # for coreml quantization

      # extras --------------------------------------
      thop  # FLOPS computation
      pycocotools>=2.0  # COCO mAP
      
usage
--------------------------------------------------------
      # content ----------------------------------------------
      有修改並增加功能的部分
      yolov5-master-> utils -> detect_yolo_fir.py
                               plots.py
                               temperature.py

detect_yolo_fir.py
------------------------------------------------------

      sys.path.append('D:\python\yolov5_notebook\yolov5-master\yolov5-master')
      其中需要修改路徑
      self.model = hub.custom(path_or_model=r'\python\yolov5_notebook\yolov5-master\yolov5-master\yolov5s.pt') 
      其中的path_or_model需要修改路徑

    def detect_yolo_big (self, img): # 輸出最大面積的搜索框
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

    def detect_yolo (self, img):  # 把搜索框位置從 tensor 轉成 數值 輸出值
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
    
      def center(xyxy): # 計算搜索框中心
          x1,y1,x2,y2 = xyxy
          x = (x1 + x2)/2
          y = (y1 + y2)/2
          center = x , y
          return center

plots.py
-------------------------------------------------

      def plot_one_box_fir(x, img, img_r, M, T, color=None, label=None, line_thickness=None): #秀出搜索框 物件名稱 體溫 祝賀語
          blessing = ["You can do it.","I have faith in you.","You're looking sharp!","You're so smart.","You're awesome !",
                      "You did a good job."," You're very professional."]
          # Plots one bounding box on image img
          tl = line_thickness # or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
          color = color or [random.randint(0, 255) for _ in range(3)]
          c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
          # img = img[c1[1] : c2[1], c1[0] : c2[0]]
          c1_r, c2_r = tp.Ptransform(c1, c2, M)

          cv2.rectangle(img, c1, c2, color, tl, cv2.LINE_AA)
          if label:
              tf = max(tl - 1, 1)  # font thickness
              imCrop = img_r[c1_r[0] : c2_r[0], c1_r[1] : c2_r[1]]
              # print("imCrop",imCrop)
              isperson = label[:6]
              # print("label:",isperson)
              if isperson == 'person':
                  # print(imCrop)
                  # if imCrop.shape[0] and imCrop.shape[1] != 0:
                      # cv2.imshow("imCrop",imCrop)
                  tep_color = tp.get_temperature(imCrop)
                  label = label +" , Tc = "+str(tep_color)



              t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
              c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
              if isperson == 'person':
                  t_size_b = cv2.getTextSize(blessing[T], 0, fontScale=tl / 3, thickness=tf)[0]
                  c3 = c1[0] + max(t_size_b[0],t_size[0]), c2[1] - t_size_b[1] - 6
                  cv2.rectangle(img, c1, c3, color, -1, cv2.LINE_AA)
                  cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                  cv2.putText(img, blessing[T], (c1[0], c1[1] - 6-t_size_b[1]), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
              else:
                  cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                  cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
              # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        
        
temperature.py
-----------------------------------

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
      def face_dect(image):  # Haar臉部辨識，沒採用
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

      def get_temperature(image): # 計算熱像儀上影像溫度
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



      def Ptransform(rect_up,rect_down,M): # 影像透視變換，因為熱像儀解析度240P而相機解析度為1080P，無法直接對使用由相機影像抓出的搜索框位置
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
