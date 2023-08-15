import cv2
import numpy as np
import os
import socket
import time
import math


def nothing(x):
    pass

#check for 2D matrix
def checkPoints(event, x, y, flags, param) :
    if event == cv2.EVENT_LBUTTONDOWN :
        print(f'current point of {param} (x, y) : ', x, y)

def vision_set():
    # while True:

    global mapx0, mapx1, mapy0, mapy1, mask0, mask1, cap0, cap1
    global ball_3D_temp, ball_3D
    global ball_cam0, ball_cam1
    global speed_tm
    global centerX, centerY

    ball_cam0 = np.array([0, 0])
    ball_cam1 = np.array([0, 0])

    ret_0, frame_0 = cap0.read()
    ret_1, frame_1 = cap1.read()

    cv2.imshow('src_0', frame_0)
    src_0 = cv2.remap(frame_0, mapx0, mapy0, cv2.INTER_LINEAR)
    src_0 = cv2.copyTo(src_0, mask0)

    cv2.imshow('src_1', frame_1)
    src_1 = cv2.remap(frame_1, mapx0, mapy0, cv2.INTER_LINEAR)
    src_1 = cv2.copyTo(src_1, mask1)

    cv2.imshow('src_0', src_0)
    cv2.imshow('src_1', src_1)

    #show current 3D points through mouse click
    # cv2.imshow('current_point0', src_0) 
    # cv2.imshow('current_point1', src_1)

    # cv2.setMouseCallback('current_point0', print_3D, 0)
    # cv2.setMouseCallback('current_point1', print_3D, 1)

    cv2.setMouseCallback('src_0', checkPoints, 0)
    cv2.setMouseCallback('src_1', checkPoints, 1)

    src_hsv_0 = cv2.cvtColor(src_0, cv2.COLOR_BGR2HSV)
    src_hsv_1 = cv2.cvtColor(src_1, cv2.COLOR_BGR2HSV)
    
    # cv2.setMouseCallback('src_0', checkPoints)


    # Detecting Color Setting
    # dst_1 = cv2.inRange(src_hsv_1, (hmin_1, smin_1, vmin_1), (hmax_1, smax_1, vmax_1))

    # cv2.imshow('dst_0', dst_0)
    # cv2.imshow('dst_1', dst_1)

    # MORPH 함수 이용하여 정확도 향상(Value Optimization)
    kernel = np.ones((3, 3), np.uint8)
    # dst_0 = cv2.morphologyEx(dst_0, cv2.MORPH_OPEN, kernel)
    # dst_0 = cv2.morphologyEx(dst_0, cv2.MORPH_CLOSE, kernel)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)

    # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득
    dst_0 = cv2.inRange(src_hsv_0, (hmin_0, smin_0, vmin_0), (hmax_0, smax_0, vmax_0))
    dst_1 = cv2.inRange(src_hsv_1, (hmin_1, smin_1, vmin_1), (hmax_1, smax_1, vmax_1))
    
    img_result_0 = cv2.bitwise_and(src_0, src_0, mask=dst_0)
    img_result_1 = cv2.bitwise_and(src_1, src_1, mask=dst_1)

    numOfLabels_0, img_label_0, stats_0, centroids_0 = cv2.connectedComponentsWithStats(dst_0)
    numOfLabels_1, img_label_1, stats_1, centroids_1 = cv2.connectedComponentsWithStats(dst_1)

    # centroids==무게중심 좌표(x,y)

    for idx, centroid in enumerate(centroids_0):
        if stats_0[idx][0] == 0 and stats_0[idx][1] == 0:
            continue

        if np.any(np.isnan(centroid)):
            continue

        x, y, width, height, area = stats_0[idx]
        centerX, centerY = int(centroid[0]), int(centroid[1])
        # print(centerX, centerY)

        if 100 < area < 1000:
            # 일정 범위 이상 & 이하인 부분에 대해서만 centroids 값 반환

            cv2.circle(src_0, (centerX, centerY), 10, (0, 0, 255), 10)
            cv2.rectangle(src_0, (x, y), (x + width, y + height), (0, 0, 255))
            ball_cam0 = np.array([centroid[0], centroid[1]], dtype=float)



    for idx, centroid in enumerate(centroids_1):

        if stats_1[idx][0] == 0 and stats_1[idx][1] == 0:
            continue

        if np.any(np.isnan(centroid)):
            continue

        x, y, width, height, area = stats_1[idx]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if 100 < area < 1000:
            # 일정 범위 이상 & 이하인 부분에 대해서만 centroids 값 반환
            cv2.circle(src_1, (int(centerX), int(centerY)), 10, (0, 0, 255), 10)
            cv2.rectangle(src_1, (x, y), (x + width, y + height), (0, 0, 255))
            ball_cam1 = np.array([centroid[0], centroid[1]], dtype=float)


    # Display

    cv2.imshow('src_0', src_0)
    # cv2.imshow('dst_0', dst_0)
    # cv2.imshow('img_result_0', img_result_0)

    cv2.imshow('src_1', src_1)
    # cv2.imshow('dst_1', dst_1)
    # cv2.imshow('img_result_1', img_result_1)

    speed_tm=cv2.TickMeter()
#   
    # sparsePrint('')
    # sparsePrint('-----------------------------------------')
    # sparsePrint(ball_cam1)
    



def predict():
    global ball_array
    global z_array
    global ball_cam1, ball_cam0
    global centerX, centerY
    global temp_0
    global slope, slope_z, deg_send #only 'slope' is used
    global x_p
    global j
    global impact
    global pcnt
    global X, Z, v_x, t
    global speed_tm

   

    #the condition at which the ball is going over the net (temp_0==1)
    # print("temp_0 : ", temp_0)

    
    # if temp_0 == 1 and centerY > 200:
    # ball_array [0,0],[0,0] --> ball_array[0]: temp_point, ball_array[1]: curr_point

    if ball_array[0][0]!=ball_cam0[0] and ball_array[0][1]!=ball_cam0[1]:
        ball_array.append([ball_cam0[0], ball_cam0[1]])
        ball_array.pop(0)
            
    if ball_array[1][1] - 647 !=0:
        # print("center_x", centerX)
        # print("center_y", centerY)
        # print("ball_array_x", ball_array[1][0])
        # print("ball_array_y", ball_array[1][1])
        if ball_array[1][1] - ball_array[0][1]!=0:
            slope = (ball_array[1][0] - ball_array[0][0]) / (ball_array[1][1] - ball_array[0][1])
        else:
            print("denominator is zero")
        # deg_send = math.degrees(math.atan(-slope))
        # deg_0 =f"{0},{0}"
        # data= f"{1},{deg_send}"
        # print("deg_send : ", deg_send)
    else:
        slope=0

    
    # x_p = slope * 24 + 0
    j=0
    x_p = (slope * (-230 - ball_array[1][1])  + ball_array[1][0] - 580) * 0.35

    #slope redefine based on x_p
    deg_send = x_p*(5/11)
    deg_0 =f"{0},{0}"
    data= f"{1},{deg_send}"
    print("deg_send : ", deg_send)



    #z 좌표 범위 지정
    
    if ball_cam1[0]==0 and ball_cam1[0]==0:
        z_array=[]

    if (ball_cam1[0]!=0 and ball_cam1[1]!=0) and (-len(z_array)==0 or (len(z_array)==1 and z_array[0][0] != ball_cam1[0] and z_array[0][1]!=ball_cam1[1])):
        if speed_tm.getAvgTimeMilli()>0:
            speed_tm.stop()
        
        tm_diff=speed_tm.getTimeMilli()

        z_array.append([ball_cam1[0], ball_cam1[1]])
        if len(z_array)==2 and z_array[1][1] - z_array[0][1] > 0:
            z_array.pop(0)

        if len(z_array)==2 and z_array[1][1] - z_array[0][1] < 0:
            X = 970 - z_array[0][0]
            if tm_diff!=0:
                v_x = (z_array[1][0] - z_array[0][0]) / tm_diff
                t = X / v_x
                Z = X*(z_array[1][1] - z_array[0][1]) / (z_array[1][0] - z_array[0][0]) - 9.8*(X**2) / 2*((v_x)**2)


        speed_tm.reset()
        speed_tm.start()
            



    
       




    if impact==1 and cnt > 0 and ball_array[1][1]-ball_array[0][1]<0:
        print("impact detection succeeded")
        print("slope: ", slope)
        print("center_x", centerX)
        print("center_y", centerY)
        print("ball_array_x1", ball_array[0][0])
        print("ball_array_y1", ball_array[0][1])
        print("ball_array_x2", ball_array[1][0])
        print("ball_array_y2", ball_array[1][1])
        print("current x" , ball_array[1][0])
        print("result x_p: ", x_p)
        
        #predicted x position
        udp_socket.sendto(str(x_p).encode(), (ip_address, 9999))
        time.sleep(0.03)
        #impact, degree
        udp_socket.sendto(data.encode(), (ip_address, 3333)) 
        
        #set deg to zero (set to initial)
        udp_socket.sendto(deg_0.encode(), (ip_address, 3333))
        
    print("result z_p: ", Z)
    print("z_array: ", z_array)
    print("X: ", X)
    
# ---------------------------------------------------y_p calc-----------------------------------------------------------

    if x_p > 55:
        x_p = 55
    elif x_p < -55:
        x_p = -55

# ----------------------------------------------------Step Calc---------------------------------------------------------

    # if 0 < abs(slope) < 0.04:
    #     step = 1

    # elif 0.04 < abs(slope) < 0.08:
    #     step = 2

    # elif 0.08 < abs(slope) < 0.12:
    #     step = 3

    # elif 0.12 < abs(slope) < 0.16:
    #     step = 4

    # else:
    #     step = 5

    # if -470 <= y_p < -200:
    #     y_p = -380

    # elif -200 <= y_p < 200:
    #     y_p = -25

    # elif 200 <= y_p <= 470:
    #     y_p = 380

# -----------------------------------------------------print------------------------------------------------------------

 
    # print('predict_result')
    # print("x_p :",x_p) 
    # print(slope)
    # print((5000+int(-y_p))*10000+step*1000+0)

# ---------------------------------------------------Data Send----------------------------------------------------------
    #data = str(x_p) #1000 부분을 조절해서, y를 맞춰야함
    # data=str(0) #fix well for good clear x_p

    #if impact == 1 and cnt > 0:
     #   udp_socket.sendto(data.encode(), (ip_address, 9999))

     #  udp_socket.sendto(str(impact).encode(), (ip_address, 3333))  # 강민석이 단거임
#
 #       udp_socket.sendto(str(0).encode(), (ip_address, 3333))



# elif temp_0 == 1 and ball_3D[1] > 11.5:
#     temp_0 = 0

#print the text sparsely so that research can read the log simultaneously.
def sparsePrint(*texts):
    print_std=20
    
    if print_std%10==0:
        for text in texts:
            print(text, end="")
        print_std=0

    print_std+=1

def reset_params():
    global curr_p, prev_p
    global slope_temp, slope
    global temp_0
    global ball_array
    global x_p

    impact = 0
    ball_array = []
    z_array = []
    temp_0 = 1
    slope = 0
    slope_temp = 0
    curr_p=[0,0]
    prev_p=[0,0]
    data_reset = str(0)
    udp_socket.sendto(data_reset.encode(), (ip_address, 9999))
    print('reset!')


if __name__ == '__main__':
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    os.chdir('C:/Users/User/Desktop/Dev/tabletennis_robot')

# -----------------------------------------------초기값 UDP Send---------------------------------------------------------

    ip_address="172.17.27.22"
    data_zero = str(0)
    udp_socket.sendto(data_zero.encode(), (ip_address, 9999))
    data_impact = str(0)
    udp_socket.sendto(data_impact.encode(), (ip_address, 3333))

    # Set Global Variables

    global hmin_0, hmax_0, smin_0, smax_0, vmin_0, vmax_0
    global hmin_1, hmax_1, smin_1, smax_1, vmin_1, vmax_1

    global impact
    global centerX, centerY
    global cnt, pcnt

    #set initial color range
    lower_color = [0, 87, 89]
    upper_color = [63, 255, 200]


    [hmin_0, smin_0, vmin_0]=lower_color
    [hmax_0, smax_0, vmax_0]=upper_color

    [hmin_1, smin_1, vmin_1]=lower_color
    [hmax_1, smax_1, vmax_1]=upper_color


    temp_0 = 1
    y_p = 0
    slope = 0
    slope_temp = 0
    centerY=480
    centerX=760
    ball_array = [[0,0],[0,0]]
    z_array = [[0,0],[0,0]]
    impact=0
    pcnt = 0
    cnt = 2
    Z=0
    X=0


    i_main = 0
    h, w = np.array([720, 1280])

    # Set Camera Matrix

    #R0 = np.linalg.inv(np.array([[-0.6403, -0.6730, -1.4113], 
     #                            [-0.5477, -0.5929, -1.4882], 
      #                           [-0.4374, -0.4086 , -1.4703]]))
    r0 = np.array([-0.06858176, 1.389907, 2.78486924])
    r1 = np.array([-0.02972071, 0.48175457, 3.08257579])

    R0, _ = cv2.Rodrigues(r0)
    R1, _ = cv2.Rodrigues(r1)
    
    T0 = np.array([6.27299391, 4.01877434, 24.29342169])
    T1 = np.array([2.15020573, 17.54851896, 30.14949137])

    # Translation Matrix between each cam & World Coord
    # Focal length of each cam

    cam0_f = np.array([419.4296, 384.6875])
    cam1_f = np.array([745.99261893, 748.10811003])

    # Principle Point of each cam

    cam0_c = np.array([647.8114, 358.0928])
    cam1_c = np.array([625.02714628, 315.57402011])

    # Intrinsics Matrix

    cam0_int = np.array([[814.49848129, 0., 568.49302368], [0., 805.90235641, 369.59529032], [0., 0., 1.]])
    cam1_int = np.array([[745.99261893, 0.,  625.02714628], [0., 748.10811003, 315.57402011], [0., 0., 1.]])
    
    mtx0 = cam0_int
    mtx1 = cam1_int

    #hstack: 가로로 두 array 붙이는 연산
    dist0 = np.array([0.3166118, -0.49218699, -0.0046719, -0.03840587, 0.25442361])
    dist1 = np.array([1.52562640e-01, -4.31254941e-01, -4.27185613e-02,  2.25366445e-04, -2.04619580e-01])

    print('intrinsics Matrix')
    print('')
    print(mtx0)
    print(dist0)
    print("")
    print(mtx1)
    print(dist1)

    # Calibration for new camera matrix

    newcameraMtx0, roi0 = cv2.getOptimalNewCameraMatrix(cam0_int, dist0, (w, h), 1, (w, h))
    print(newcameraMtx0)
    print("")
    newcameraMtx1, roi1 = cv2.getOptimalNewCameraMatrix(cam1_int, dist1, (w, h), 1, (w, h))
    print(newcameraMtx1)

    print(roi0)
    print(roi1)

    # T0 = np.array([0,0,0]) # Translation vector
    RT0 = np.zeros((3, 4))  # combined Rotation/Translation matrix
    RT0[:3, :3] = R0
    RT0[:3, 3] = T0
    P0 = np.dot(newcameraMtx0, RT0)  # Projection matrix

    # # define pose 1

    # T1 = np.array([0,0,2.])
    RT1 = np.zeros((3, 4))
    RT1[:3, :3] = R1
    RT1[:3, 3] = T1
    P1 = np.dot(newcameraMtx1, RT1)

    print(P0)
    print(P1)

    
    mapx0, mapy0 = cv2.initUndistortRectifyMap(mtx0, dist0, None, newcameraMtx0, (w, h), 5)
    mapx1, mapy1 = cv2.initUndistortRectifyMap(mtx1, dist1, None, newcameraMtx1, (w, h), 5) 

    # CAP_DSHOW 가 그냥 Index Calling에 비해 속도 훨씬 빠름

    # p1 = Process(target=vision_set())
    # p2 = Process(target=predict())

    cap0 = cv2.VideoCapture(cv2.CAP_DSHOW + 0)
    cap1 = cv2.VideoCapture(cv2.CAP_DSHOW + 1)

    cap0.isOpened()
    cap1.isOpened()

    # Camera0_Setting
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap0.set(cv2.CAP_PROP_FRAME_COUNT, 60)
    cap0.set(cv2.CAP_PROP_POS_MSEC, 11)
    cap0.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap0.set(cv2.CAP_PROP_FPS, 90)
    cap0.set(cv2.CAP_PROP_EXPOSURE, -7)

    w_0 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_0 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("the width and height of the CAM0: ", w_0, h_0)

    # Camera1_Setting
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap1.set(cv2.CAP_PROP_FRAME_COUNT, 60)
    cap1.set(cv2.CAP_PROP_POS_MSEC, 11)
    cap1.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap1.set(cv2.CAP_PROP_FPS, 90)
    cap1.set(cv2.CAP_PROP_EXPOSURE, -7)
    # cap1.set(cv2.CAP_PROP_BRIGHTNESS, 500)
    print("the cap1 fps: ", cap1.get(cv2.CAP_PROP_FPS))

    w_0 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_0 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("the width and height of the CAM1: ", w_0, h_0)

    mask0 = cv2.imread('cam0_mask_cali_v2.jpg', cv2.IMREAD_GRAYSCALE)
    mask1 = cv2.imread('cam1_mask_cali_v2.jpg', cv2.IMREAD_GRAYSCALE)

    # cv2.namedWindow('src')

    # cv2.namedWindow('dst_0') #dst_0 is the mask(gray scale) of the ball
    # cv2.namedWindow('dst_1')


    # 연산 시간 측정
    tm = cv2.TickMeter()
    
    # the standard for printing current state
    cnt = 2
    pcnt = 0

    while True:

        tm.reset()

        tm.start()
        vision_set()

        if cv2.waitKey(1) & 0xFF == ord('r'):
            reset_params()
        elif cv2.waitKey(1) & 0xFF == 27:
            print('break!')
            break


        if ball_cam0[1] < 150 and [ball_cam0[0], ball_cam0[1]]!=[0,0]: #maybe std at which the robot should impact
            impact = 1
            cnt = cnt - 1
        else :
            impact = 0
            cnt = 2

        predict()

        
        sparsePrint("centerX: ", centerX)
        sparsePrint("centerY: ", centerY)

    
        sparsePrint("impact: ", impact)

        sparsePrint("cnt : ", cnt)
        


        # if print_std%print_now==0:
        #     print("temp_0: (ignored)", temp_0)

        tm.stop()
        
        sparsePrint('Calc time : {}ms.'.format(tm.getTimeMilli()))
         


    cv2.destroyAllWindows()
    cap1.release()
