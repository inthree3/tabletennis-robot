import numpy as np
import cv2

num=0
cap=cv2.VideoCapture(num)

ret,frame=cap.read()

cv2.imwrite(f'cam{num}_ex.jpg', frame)

cap.release()
cv2.destroyAllWindows()
