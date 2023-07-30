import numpy as np
import cv2

num=0
cap=cv2.VideoCapture(num)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret,frame=cap.read()

cv2.imwrite(f'cam{num}_ex.jpg', frame)

cap.release()
cv2.destroyAllWindows()
