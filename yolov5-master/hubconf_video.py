"""File for accessing YOLOv5 via PyTorch Hub https://pytorch.org/hub/

Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, channels=3, classes=80)
"""

from pathlib import Path

import torch

from models.yolo import Model
from utils.general import set_logging
from utils.google_utils import attempt_download
import cv2
import time

dependencies = ['torch', 'yaml']
set_logging()


def create(name, pretrained, channels, classes):
    """Creates a specified YOLOv5 model

    Arguments:
        name (str): name of model, i.e. 'yolov5s'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes

    Returns:
        pytorch model
    """
    config = Path(__file__).parent / 'models' / f'{name}.yaml'  # model.yaml path
    try:
        model = Model(config, channels, classes)
        if pretrained:
            fname = f'{name}.pt'  # checkpoint filename
            attempt_download(fname)  # download if not found locally
            # ckpt = torch.load(fname, map_location=torch.device('cpu'))  # load
            ckpt = torch.load(fname, map_location=torch.device('cuda'))  # load
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            state_dict = {k: v for k, v in state_dict.items() if model.state_dict()[k].shape == v.shape}  # filter
            model.load_state_dict(state_dict, strict=False)  # load
            if len(ckpt['model'].names) == classes:
                model.names = ckpt['model'].names  # set class names attribute
            # model = model.autoshape()  # for PIL/cv2/np inputs and NMS
        return model

    except Exception as e:
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = 'Cache maybe be out of date, try force_reload=True. See %s for help.' % help_url
        raise Exception(s) from e


def yolov5s(pretrained=False, channels=3, classes=80):
    """YOLOv5-small model from https://github.com/ultralytics/yolov5

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov5s', pretrained, channels, classes)


def yolov5m(pretrained=False, channels=3, classes=80):
    """YOLOv5-medium model from https://github.com/ultralytics/yolov5

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov5m', pretrained, channels, classes)


def yolov5l(pretrained=False, channels=3, classes=80):
    """YOLOv5-large model from https://github.com/ultralytics/yolov5

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov5l', pretrained, channels, classes)


def yolov5x(pretrained=False, channels=3, classes=80):
    """YOLOv5-xlarge model from https://github.com/ultralytics/yolov5

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov5x', pretrained, channels, classes)


def custom(path_or_model='path/to/model.pt'):
    """YOLOv5-custom model from https://github.com/ultralytics/yolov5

    Arguments (3 options):
        path_or_model (str): 'path/to/model.pt'
        path_or_model (dict): torch.load('path/to/model.pt')
        path_or_model (nn.Module): torch.load('path/to/model.pt')['model']

    Returns:
        pytorch model
    """
    model = torch.load(path_or_model, map_location=torch.device('cuda')) if isinstance(path_or_model, str) else path_or_model  # load checkpoint
    # model = torch.load(path_or_model) if isinstance(path_or_model, str) else path_or_model  # load checkpoint
    if isinstance(model, dict):
        model = model['model']  # load model

    hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
    print(model.parameters())
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    hub_model.names = model.names  # class names
    return hub_model


if __name__ == '__main__':
    # model = create(name='yolov5s', pretrained=True, channels=3, classes=80)  # pretrained example
    # model = custom(path_or_model='path/to/model.pt')  # custom example
    model = custom(path_or_model=r'runs\train\exp\weights\best.pt') 
    
    model = model.autoshape()  # for PIL/cv2/np inputs and NMS

    # Verify inference
    # from PIL import Image

    # imgs = [Image.open(x) for x in Path('data/images').glob('*.jpg')]
    
    cap = cv2.VideoCapture(0)
    # cap.set(cv2. CAP_PROP_FRAME_WIDTH, 2560)
    # cap.set(cv2. CAP_PROP_FRAME_HEIGHT, 1980)

    while(True):
        # 從攝影機擷取一張影像
        st = time.time()
        fps = cap.get(cv2.CAP_PROP_FPS)
        # time.sleep(0.1)
        print(fps)
        ret, frame = cap.read()
        
        results = model(frame)
        pos = results.xyxy[0].cpu().numpy()
        et = time.time()
        print(et-st)
        # pos = pos[0,0:4]
        # print("找尋位子",pos)
        # 顯示圖片
        cv2.imshow('frame', frame)
    
        # 若按下 q 鍵則離開迴圈
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 釋放攝影機
    cap.release()
    
    # 關閉所有 OpenCV 視窗
    cv2.destroyAllWindows()

    # # print(imgs)
    # results = model(imgs)
    # print(results.xyxy)
    # print("h1",results.xyxy[0])
    # print("h2",results.xyxy[1])
    # # results.show()
    # # results.print()

    # results = model(imgs)
    # print(results.xyxy)
    # print("h11",results.xyxy[0])
    # print("h22",results.xyxy[1])
    # # results.show()
    # # results.print()