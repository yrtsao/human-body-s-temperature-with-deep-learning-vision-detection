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
