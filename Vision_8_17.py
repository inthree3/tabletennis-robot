import cv2
import numpy as np
import os
import socket


def nothing(x):
    pass


#function: set the ball color range through clicking
def click_color_0(event, x, y, flags, params):
    #global vars declaration
    global hmin_0, hmax_0, smin_0, smax_0, vmin_0, vmax_0
    tolerance=40

    if event == cv2.EVENT_LBUTTONDBLCLK:
        selected_color=params[y, x]
        hsv_color=cv2.cvtColor(np.uint8([[selected_color]]), cv2.COLOR_BGR2HSV)
        hue=int(hsv_color[0][0][0]) #for cv2.inRange error being resolved
        shade=int(hsv_color[0][0][1])
        value=int(hsv_color[0][0][2])
        lower_color=[hue-tolerance, shade-tolerance, value-tolerance]
        upper_color=[hue+tolerance, shade+tolerance, value+tolerance]

        #set the color range as global
        [hmin_0, smin_0, vmin_0]=lower_color
        [hmax_0, smax_0, vmax_0]=upper_color
        print("Selected HSV Range is ", lower_color, upper_color)

def click_color_1(event, x, y, flags, params):
    #global vars declaration
    global hmin_1, hmax_1, smin_1, smax_1, vmin_1, vmax_1
    tolerance=35 #set color range width

    if event == cv2.EVENT_LBUTTONDBLCLK:
        selected_color=params[y, x]
        hsv_color=cv2.cvtColor(np.uint8([[selected_color]]), cv2.COLOR_BGR2HSV)
        hue=int(hsv_color[0][0][0]) #for cv2.inRange error being resolved
        shade=int(hsv_color[0][0][1])
        value=int(hsv_color[0][0][2])
        lower_color=[hue-tolerance, shade-tolerance, value-tolerance]
        upper_color=[hue+tolerance, shade+tolerance, value+tolerance]

        #set the color range as global
        [hmin_1, smin_1, vmin_1]=lower_color
        [hmax_1, smax_1, vmax_1]=upper_color
        print("Selected HSV Range is ", lower_color, upper_color)

#function for showing the current 3D point
def print_3D(event, x, y, flags, params):
        global ball_3D
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print(f"current point of cam{params}: ", ball_3D)


def vision_set(print_std):
    # while True:

    global mapx0, mapx1, mapy0, mapy1, mask0, mask1, cap0, cap1
    global ball_3D_temp, ball_3D
    # global ball_cam0, ball_cam1

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
    cv2.imshow('current_point0', src_0)
    cv2.imshow('current_point1', src_1)

    cv2.setMouseCallback('current_point0', print_3D, 0)
    cv2.setMouseCallback('current_point1', print_3D, 1)


    src_hsv_0 = cv2.cvtColor(src_0, cv2.COLOR_BGR2HSV)
    src_hsv_1 = cv2.cvtColor(src_1, cv2.COLOR_BGR2HSV)

    cv2.setMouseCallback('src_0', click_color_0, src_hsv_0)
    cv2.setMouseCallback('src_1', click_color_1, src_hsv_1)

    #check for 2D matrix
    def checkPoints(event, x, y, flags, param) :
        if event == cv2.EVENT_LBUTTONDOWN :
            print('current (x, y) : ', x, y)

    # cv2.setMouseCallback('src_0', checkPoints)


    # Detecting Color Setting
    dst_0 = cv2.inRange(src_hsv_0, (hmin_0, smin_0, vmin_0), (hmax_0, smax_0, vmax_0))
    dst_1 = cv2.inRange(src_hsv_1, (hmin_1, smin_1, vmin_1), (hmax_1, smax_1, vmax_1))

    # cv2.imshow('dst_0', dst_0)
    # cv2.imshow('dst_1', dst_1)

    # MORPH 함수 이용하여 정확도 향상(Value Optimization)
    # kernel = np.ones((3, 3), np.uint8)
    # dst_0 = cv2.morphologyEx(dst_0, cv2.MORPH_OPEN, kernel)
    # dst_0 = cv2.morphologyEx(dst_0, cv2.MORPH_CLOSE, kernel)
    # dst_1 = cv2.morphologyEx(dst_1, cv2.MORPH_OPEN, kernel)
    # dst_1 = cv2.morphologyEx(dst_1, cv2.MORPH_CLOSE, kernel)

    # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득

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

        if 100 < area < 8000:
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

        if 100 < area < 8000:
            # 일정 범위 이상 & 이하인 부분에 대해서만 centroids 값 반환
            cv2.circle(src_1, (centerX, centerY), 10, (0, 0, 255), 10)
            cv2.rectangle(src_1, (x, y), (x + width, y + height), (0, 0, 255))
            ball_cam1 = np.array([centroid[0], centroid[1]], dtype=float)

    if ball_cam0[0] != 0 and ball_cam1[0] != 0:

        ball_tri = np.array(cv2.triangulatePoints(P0, P1, ball_cam0, ball_cam1))
        ball_3D = ball_tri[:3] / ball_tri[-1]

    else:

        ball_3D = ball_3D_temp

    # Display

    cv2.imshow('src_0', src_0)
    cv2.imshow('dst_0', dst_0)
    cv2.imshow('img_result_0', img_result_0)

    cv2.imshow('src_1', src_1)
    cv2.imshow('dst_1', dst_1)
    cv2.imshow('img_result_1', img_result_1)

    if print_std%5==0:
        print('')
        print('-----------------------------------------')
        print(ball_cam0)
        print(ball_cam1)
        print('ball_3D_temp')
        print(ball_3D_temp)
        print('ball_3D')
        print(ball_3D)


def predict():

    global ball_array
    global temp_0
    global ball_3D, ball_3D_temp
    global slope, slope_temp
    global y_p

    if temp_0 == 1 and ball_3D[1] > :
        ball_array[:, 0:1] = ball_3D

        slope = (ball_array[1, 0] - 0) / (ball_array[0, 0] - 150)
        y_p = slope * (2500 - 150) + 0

# ---------------------------------------------------y_p calc-----------------------------------------------------------

        if y_p > 470:
            y_p = 470

        elif y_p < -470:
            y_p = -470

# ----------------------------------------------------Step Calc---------------------------------------------------------

        if 0 < abs(slope) < 0.04:
            step = 1

        elif 0.04 < abs(slope) < 0.08:
            step = 2

        elif 0.08 < abs(slope) < 0.12:
            step = 3

        elif 0.12 < abs(slope) < 0.16:
            step = 4

        else:
            step = 5

        if -470 <= y_p < -200:
            y_p = -380

        elif -200 <= y_p < 200:
            y_p = -25

        elif 200 <= y_p <= 470:
            y_p = 380

# -----------------------------------------------------print------------------------------------------------------------

        print(ball_array[:, 0])
        print(ball_array[:, 1])
        print('predict_result')
        print(y_p)
        print(slope)
        print((5000+int(-y_p))*10000+step*1000+0)

# ---------------------------------------------------Data Send----------------------------------------------------------
        data = str((5000 + int(-y_p)) * 10000 + step * 1000 + 0)
        udp_socket.sendto(data.encode(), (ip_address, 9999))

    if temp_0 == 1 and ball_3D[1] > 150:
        temp_0 = 0


def reset_params():

    global ball_3D, i, ball_3D_temp
    global slope_temp, slope
    global temp_0
    global ball_array

    ball_array = np.zeros((3, 2))
    temp_0 = 1
    ball_3D = np.zeros((3, 1))
    ball_3D_temp = np.zeros((3, 1))
    slope = 0
    slope_temp = 0
    data_reset = str(50000000)
    udp_socket.sendto(data_reset.encode(), (ip_address, 9999))
    print('reset!')


if __name__ == '__main__':

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    os.chdir('C:/Users/User/Desktop/Dev/tabletennis_robot')

# -----------------------------------------------초기값 UDP Send---------------------------------------------------------

    ip_address="172.17.27.22"
    data_zero = str(50000000)
    udp_socket.sendto(data_zero.encode(), (ip_address, 9999))
    data_impact = str(0)
    udp_socket.sendto(data_impact.encode(), (ip_address, 3333))

    # Set Global Variables

    global hmin_0, hmax_0, smin_0, smax_0, vmin_0, vmax_0
    global hmin_1, hmax_1, smin_1, smax_1, vmin_1, vmax_1

    #set initial color range
    lower_color = [18, 71, 100]
    upper_color = [25, 88, 100]

    [hmin_0, smin_0, vmin_0]=lower_color
    [hmax_0, smax_0, vmax_0]=upper_color

    [hmin_1, smin_1, vmin_1]=lower_color
    [hmax_1, smax_1, vmax_1]=upper_color


    temp_0 = 1
    i = 0
    y_p = 0
    slope = 0
    slope_temp = 0
    ball_array = np.zeros((3, 2))

    i_main = 0
    ball_3D_temp = np.zeros((3, 1))
    ball_3D = np.zeros((3, 1))
    h, w = np.array([720, 1280])

    # Set Camera Matrix

    #R0 = np.linalg.inv(np.array([[-0.6403, -0.6730, -1.4113], 
     #                            [-0.5477, -0.5929, -1.4882], 
      #                           [-0.4374, -0.4086 , -1.4703]]))
    r0 = np.array([-0.08196, 0.41632342, 3.10042661])
    r1 = np.array([-0.06858176, 1.389907, 2.78486924])

    R0, _ = cv2.Rodrigues(r0)
    R1, _ = cv2.Rodrigues(r1)
    
    T0 = np.array([1.48079843, 6.07775434, 24.29342169])
    T1 = np.array([6.27299391, 4.01877434, 24.29342169])

    # Translation Matrix between each cam & World Coord
    # Focal length of each cam

    cam0_f = np.array([535.90786742, 530.60972232])
    cam1_f = np.array([419.4296, 384.6875])

    # Principle Point of each cam

    cam0_c = np.array([655.77404621, 354.47028656])
    cam1_c = np.array([647.8114, 358.0928])

    # Distortion

    cam0_dist_r = np.array([-0.1408, 0.0091])
    cam0_dist_t = np.array([0, 0])
    cam1_dist_r = np.array([-1.0359, -2.1435])
    cam1_dist_t = np.array([0, 0])

    # Intrinsics Matrix

    cam0_int = np.array([[783.89487299, 0, 673.87675856], [0, 787.96916992, 388.49312521], [0, 0, 1]])
    cam1_int = np.array([[784.08203396, 0., 595.52834178], [0., 781.30731967, 365.24995514], [0., 0., 1.]])

    mtx0 = cam0_int
    mtx1 = cam1_int

    dist0 = np.array([0.2724565, -0.65692629, 0.01741777, 0.00634816, 0.37098618]) #hstack: 가로로 두 array 붙이는 연산
    dist1 = np.array([0.27610058, -0.53822435, -0.00619961, -0.01892479, 0.31669127])

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
    cap0.set(cv2.CAP_PROP_FRAME_COUNT, 60)
    cap0.set(cv2.CAP_PROP_POS_MSEC, 11) #set fps approx 90
    cap0.set(cv2.CAP_PROP_AUTOFOCUS, 0) #turn-off autofocus function
    cap0.set(cv2.CAP_PROP_FPS, 90)
    # cap0.set(cv2.CAP_PROP_EXPOSURE, -6)
    # cap0.set(cv2.CAP_PROP_BRIGHTNESS, 500)

    w_0 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_0 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("the width and height of the CAM0: ", w_0, h_0)

    # Camera1_Setting

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap1.set(cv2.CAP_PROP_FRAME_COUNT, 60)
    cap1.set(cv2.CAP_PROP_POS_MSEC, 11)
    cap1.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap1.set(cv2.CAP_PROP_FPS, 90)
    # cap1.set(cv2.CAP_PROP_EXPOSURE, -6)
    # cap1.set(cv2.CAP_PROP_BRIGHTNESS, 500)

    w_1 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_1 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("the width and height of the CAM1: ", w_1, h_1)

    mask0 = cv2.imread('cam0_mask_cali_v2.jpg', cv2.IMREAD_GRAYSCALE)
    mask1 = cv2.imread('cam1_mask_cali_v2.jpg', cv2.IMREAD_GRAYSCALE)

    # cv2.namedWindow('src')

    # cv2.namedWindow('dst_0') #dst_0 is the mask(gray scale) of the ball
    # cv2.namedWindow('dst_1')


    # 연산 시간 측정
    tm = cv2.TickMeter()
    
    # the standard for printing current state
    print_std=0
    print_now=5

    while True:

        tm.reset()
        tm.start()
        vision_set(print_std)

        if (ball_3D[1] - ball_3D_temp[1]) > 0:
            predict()

        if cv2.waitKey(1) & 0xFF == ord('r'):
            reset_params()
        elif cv2.waitKey(1) & 0xFF == 27:
            print('break!')
            break

        if ball_3D[1] >= 12: #maybe std at which the robot should impact
            impact = 1
        else:
            impact = 0

        if print_std%print_now==0:
            print('impact')
            print(impact)
        data_impact = str(impact)
        udp_socket.sendto(data_impact.encode(), (ip_address, 3333))

        ball_3D_temp = ball_3D

        if print_std%print_now==0:
            print(temp_0)

        tm.stop()
        if print_std%print_now==0:
            print('Calc time : {}ms.'.format(tm.getTimeMilli()))
            print_std=0

        print_std+=1

    cv2.destroyAllWindows()
    cap0.release()
    cap1.release()
