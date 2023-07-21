import cv2
import numpy as np
cap=cv2.VideoCapture(1)

i=0
while True:
    ret, frame=cap.read()
    if not ret:
        break

    #convert the color from bgr to hsv
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #range of orange color
    lower_color = np.array([10, 100, 100])
    upper_color = np.array([25, 255, 255])
    
    #generate the mask
    mask = cv2.inRange(hsv, lower_color, upper_color)

    #from mask, generating the contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        perimeter=cv2.arcLength(contour, True)

        approx=cv2.approxPolyDP(contour, 0.04*perimeter, True)

        #calculating the center point
        moments=cv2.moments(approx)
        if moments["m00"]==0:
            cX = int(moments["m10"])
            cY = int(moments["m01"])
        else:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])

        circle=cv2.circle(hsv, (cX, cY), 5, (0, 255, 0), -1)
    
    if i%100==0:
        print(f"cX: {cX}, cY: {cY}")
        i=0

    cv2.imshow("frame", hsv)
    cv2.imshow("mask", mask)
    cv2.imshow("circle", circle)


    #'p' for existing the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    i+=1

cap.release()
cv2.destroyAllWindows()