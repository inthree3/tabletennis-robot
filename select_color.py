import cv2
import numpy as np
import matplotlib as plt
cap=cv2.VideoCapture(1)

i=0

# range of orange color
lower_color = np.array([18, 71, 100])
upper_color = np.array([25, 88, 100])

#define initial cX, cY
cX=-100
cY=-100

def plot_cX_cY(cX, cY):
    Xs=[]
    Ys=[]

    Xs.append(cX)
    Ys.append(cY)

    plt.plot(Xs, Ys)


#select the color through clicking
def click_color(event, x, y, flags, params):
    global lower_color, upper_color
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_color=frame[y, x]
        hsv_color=cv2.cvtColor(np.uint8([[selected_color]]), cv2.COLOR_BGR2HSV)
        hue=hsv_color[0][0][0]
        shade=hsv_color[0][0][1]
        value=hsv_color[0][0][2]
        lower_color=np.array([hue-10, shade-10, value-10])
        upper_color=np.array([hue+10, shade+10, value+10])
        print("Selected HSV Range is ",lower_color, upper_color)

while True:
    ret, frame=cap.read()
    if not ret:
        break

    #convert the color from bgr to hsv
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
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
        i+=1

    cv2.imshow("frame", hsv)
    cv2.imshow("mask", mask)
    
    cv2.setMouseCallback("frame", click_color)

    i+=1

    #'p' for existing the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()