import cv2
import numpy as np

cap=cv2.VideoCapture(1)

ball_positions=[]
#the initial color range of the ball
lower_range = np.array([0, 120, 70])
upper_range = np.array([10, 255, 255])
selected_color=False

#function for selecting color
def select_colors(event, x, y, flags, param):
    global lower_range, upper_range, selected_color
    
    if event==cv2.EVENT_LBUTTONDOWN:
        selected_color=True
        selected_pixel=hsv_frame[y, x]
        lower_range = np.array([selected_pixel[0]-10, 120, 70])
        upper_range = np.array([selected_pixel[0]+10, 255, 255])

        print("Color setting finish")

while True:
    ret, frame=cap.read()
    if not ret:
        break

    #ball detection
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if selected_color==False:
        while True:
            cv2.imshow("Select Color", hsv)
            cv2.setMouseCallback("Select Color", select_colors, hsv)

            if selected_color==True:
                cv2.destroyWindow("Select Color")
                break

    #masking to the color matched parts
    mask=cv2.inRange(hsv, lower_range, upper_range)
    
    #contour detection
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        #calculating the moment
        M=cv2.moments(contour)

        #calculating the center of the ball
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0


        ball_positions.append((cX, cY))

        #display cX and cY values
        cv2.putText(frame, f"cX: {cX}, cY: {cY}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    #'p' for existing the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(ball_positions)
cap.release()
cv2.destroyAllWindows()