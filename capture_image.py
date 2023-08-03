import numpy as np
import cv2

num=0
cap=cv2.VideoCapture(num)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret,frame=cap.read()
kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])

sharp=cv2.filter2D(frame, -1, kernel)

cv2.imwrite(f'cam{num}_ex.jpg', sharp)

cap.release()
cv2.destroyAllWindows()
