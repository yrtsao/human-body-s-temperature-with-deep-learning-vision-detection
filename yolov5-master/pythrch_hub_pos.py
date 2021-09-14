import cv2
import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).fuse().autoshape()  # for PIL/cv2/np inputs and NMS

# Images
for f in ['zidane.jpg', 'bus.jpg']:  # download 2 images
    print(f'Downloading {f}...')
    torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v1.0/' + f, f)
img1 = Image.open('zidane.jpg')  # PIL image
img2 = cv2.imread('bus.jpg')[:, :, ::-1]  # OpenCV image (BGR to RGB)
imgs = [img1, img2]  # batched list of images

# # Inference
# results = model(imgs, size=640)  # includes NMS

# # Results
# results.print()  # print results to screen
# results.show()  # display results
# results.save()  # save as results1.jpg, results2.jpg... etc.

# # # Data
# print('\n', results.xyxy[0])  # print img1 predictions
# #          x1 (pixels)  y1 (pixels)  x2 (pixels)  y2 (pixels)   confidence        class
# # tensor([[7.47613e+02, 4.01168e+01, 1.14978e+03, 7.12016e+02, 8.71210e-01, 0.00000e+00],
# #         [1.17464e+02, 1.96875e+02, 1.00145e+03, 7.11802e+02, 8.08795e-01, 0.00000e+00],
# #         [4.23969e+02, 4.30401e+02, 5.16833e+02, 7.20000e+02, 7.77376e-01, 2.70000e+01],
# #         [9.81310e+02, 3.10712e+02, 1.03111e+03, 