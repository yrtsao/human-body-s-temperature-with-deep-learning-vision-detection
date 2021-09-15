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


