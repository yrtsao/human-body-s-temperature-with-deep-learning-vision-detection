import cv2
import temperature as tp
import random

# 選擇第二隻攝影機
cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# color = [random.randint(0, 255) for _ in range(3)]
while(True):
  # 從攝影機擷取一張影像
  ret, frame = cap.read()
  # face = tp.face_dect(frame)
  # # tem = tp.get_temperature(frame)

  # print(face)
  # if face != ():
  #   (x_c,y_c,w_c,h_c) = face
  #   cv2.rectangle(frame, (x_c, y_c ), (x_c + w_c,y_c + h_c), color, 3 , cv2.LINE_AA)
  # 顯示圖片
  cv2.imshow('frame', frame)
  # tem = tp.get_temperature(frame)
  # 若按下 q 鍵則離開迴圈
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()