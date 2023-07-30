import cv2
import numpy as np
import os

# os.chdir('C:\Example\PingPong_Vision')

# Set Initial Value

global hmin_0, hmax_0, smin_0, smax_0, vmin_0, vmax_0
global hmin_1, hmax_1, smin_1, smax_1, vmin_1, vmax_1

ball_cam0 = np.array([0, 0])
ball_cam1 = np.array([0, 0])
ball_3D_temp = np.array([[0], [0], [0], [1]])

h, w = np.array([720, 1280])


#set initial color range
lower_color = [18, 71, 100]
upper_color = [25, 88, 100]

[hmin_0, smin_0, vmin_0]=lower_color
[hmax_0, smax_0, vmax_0]=upper_color

[hmin_1, smin_1, vmin_1]=lower_color
[hmax_1, smax_1, vmax_1]=upper_color


def nothing(x):
    pass

def click_color_0(event, x, y, flags, params):
    #global vars declaration
    global hmin_0, hmax_0, smin_0, smax_0, vmin_0, vmax_0


    if event == cv2.EVENT_LBUTTONDBLCLK:
        selected_color=params[y, x]
        hsv_color=cv2.cvtColor(np.uint8([[selected_color]]), cv2.COLOR_BGR2HSV)
        hue=int(hsv_color[0][0][0]) #for cv2.inRange error being resolved
        shade=int(hsv_color[0][0][1])
        value=int(hsv_color[0][0][2])
        lower_color=[hue-10, shade-10, value-10]
        upper_color=[hue+10, shade+10, value+10]

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


# Set Camera Matrix

R0 = np.linalg.inv(np.array([[0.9242, -0.2864, 0.2526],
                             [0.3812, 0.7312, -0.5658],
                             [-0.0227, 0.6192, 0.7849]]))

R1 = np.linalg.inv(np.array([[0.6989, -0.3543, 0.6213],
                             [0.7148, 0.3153, -0.6243],
                             [0.0253, 0.8804, 0.4736]]))

T0 = np.array([-1142.2, 183.7513, 1669.1])

T1 = np.array([-398.5379, 165.7630, 1398.9])

# Translation Matrix between each cam & World Coord

# Focal length(Re)
cam0_f = np.array([766.8537, 769.1458])
cam1_f = np.array([765.3281, 767.4927])

# Principle Point(Re)
cam0_c = np.array([647.6205, 345.3208])
cam1_c = np.array([641.5385, 366.9369])

# Distortion
cam0_dist_r = np.array([0.1313, -0.2132])
cam0_dist_t = np.array([0, 0])
cam1_dist_r = np.array([0.1331, -0.2226])
cam1_dist_t = np.array([0, 0])

# Intrinsics Matrix
cam0_int = np.array([[cam0_f[0], 0, cam0_c[0]], [0, cam0_f[1], cam0_c[1]], [0, 0, 1]])
cam1_int = np.array([[cam1_f[0], 0, cam1_c[0]], [0, cam1_f[1], cam1_c[1]], [0, 0, 1]])

print('intrinsics Matrix')
print('')
print(cam0_int)
print(cam1_int)

mtx0 = cam0_int
mtx1 = cam1_int
dist0 = np.hstack([cam0_dist_r, cam0_dist_t])
dist1 = np.hstack([cam1_dist_r, cam1_dist_t])
print(mtx0)
print(dist0)
print("")
print(mtx1)
print(dist1)

# calibration for new camera matrix

newcameraMtx0, roi0 = cv2.getOptimalNewCameraMatrix(mtx0, dist0, (w, h), 1, (w, h))
print(newcameraMtx0)
print("")
newcameraMtx1, roi1 = cv2.getOptimalNewCameraMatrix(mtx1, dist1, (w, h), 1, (w, h))
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

# camera fast read
cap0 = cv2.VideoCapture(cv2.CAP_DSHOW+0)
cap1 = cv2.VideoCapture(cv2.CAP_DSHOW+1)

# cap0 = cv2.VideoCapture(0)
# cap1 = cv2.VideoCapture(1)

cap0.isOpened()
cap1.isOpened()

# Camera0_Setting
cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap0.set(cv2.CAP_PROP_FRAME_COUNT, 60)
cap0.set(cv2.CAP_PROP_POS_MSEC, 11) #set fps approx 90
cap0.set(cv2.CAP_PROP_AUTOFOCUS, 0) #turn-off autofocus function

w_0 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
h_0 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(w_0, h_0)

# Camera1_Setting
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap1.set(cv2.CAP_PROP_FRAME_COUNT, 60)
cap1.set(cv2.CAP_PROP_POS_MSEC, 11)
cap1.set(cv2.CAP_PROP_AUTOFOCUS, 0)

w_1 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
h_1 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(w_1, h_1)

mask0 = cv2.imread('cam0_mask_cali_v2.jpg', cv2.IMREAD_GRAYSCALE)
mask1 = cv2.imread('cam1_mask_cali_v2.jpg', cv2.IMREAD_GRAYSCALE)

# cv2.namedWindow('src')

# cv2.namedWindow('dst_0')
# cv2.namedWindow('dst_1')

#print speed criteria
print_std=0

while True:

    ret_0, frame_0 = cap0.read()
    ret_1, frame_1 = cap1.read()

    cv2.imshow('src_0', frame_0)
    src_0 = cv2.remap(frame_0, mapx0, mapy0, cv2.INTER_LINEAR)
    src_0 = cv2.copyTo(src_0, mask0)

    cv2.imshow('src_1', frame_1)
    src_1 = cv2.remap(frame_1, mapx1, mapy1, cv2.INTER_LINEAR)
    src_1 = cv2.copyTo(src_1, mask1)

    #     # 이미지 불러와졌는지
    #     if src is None:
    #         print('Image load failed!')
    #         sys.exit()

    cv2.imshow('src_0', src_0)
    cv2.imshow('src_1', src_1)

    src_hsv_0 = cv2.cvtColor(src_0, cv2.COLOR_BGR2HSV)
    src_hsv_1 = cv2.cvtColor(src_1, cv2.COLOR_BGR2HSV)

    # Detecting Color Setting
    dst_0 = cv2.inRange(src_hsv_0, (hmin_0, smin_0, vmin_0), (hmax_0, smax_0, vmax_0))
    dst_1 = cv2.inRange(src_hsv_1, (hmin_1, smin_1, vmin_1), (hmax_1, smax_1, vmax_1))

    #dst_0 and dst_1 are mask image for detecting the balls
    # cv2.imshow('dst_0', dst_0)
    # cv2.imshow('dst_1', dst_1)

    # MORPH 함수 이용하여 정확도 향상(Value Optimization): morph가 ball mask를 없앨 수 있다고 생각해 주석 처리 (cam1)
    # kernel = np.ones((3, 3), np.uint8)
    # dst_0 = cv2.morphologyEx(dst_0, cv2.MORPH_OPEN, kernel)
    # dst_0 = cv2.morphologyEx(dst_0, cv2.MORPH_CLOSE, kernel)
    # dst_1 = cv2.morphologyEx(dst_1, cv2.MORPH_OPEN, kernel)
    # dst_1 = cv2.morphologyEx(dst_1, cv2.MORPH_CLOSE, kernel)

    # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득
    # 실제 구동시 X
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

        if 25 < area < 300:
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

        if 25 < area < 300:
            # 일정 범위 이상 & 이하인 부분에 대해서만 centroids 값 반환
            cv2.circle(src_1, (centerX, centerY), 10, (0, 0, 255), 10)
            cv2.rectangle(src_1, (x, y), (x + width, y + height), (0, 0, 255))
            ball_cam1 = np.array([centroid[0], centroid[1]], dtype=float)

    if ball_cam0[0] != 0 and ball_cam1[0] != 0:
        ball_3D = np.array(cv2.triangulatePoints(P0, P1, ball_cam0, ball_cam1))
        ball_3D = ball_3D[:3] / ball_3D[-1]
    else:
        ball_3D = ball_3D_temp

    # Display

    cv2.imshow('src_0', src_0)
    cv2.imshow('dst_0', dst_0)
    cv2.setMouseCallback('src_0', click_color_0, src_hsv_0)
    cv2.imshow('img_result_0', img_result_0)

    cv2.imshow('src_1', src_1)
    cv2.imshow('dst_1', dst_1)
    cv2.setMouseCallback('src_1', click_color_1, src_hsv_1)
    cv2.imshow('img_result_1', img_result_1)

    # Store temp value for Predicting

    ball_cam0_temp = ball_cam0
    ball_cam1_temp = ball_cam1

    ball_3D_temp = ball_3D
    if print_std%10==0:
        print('ball_3D')
        print(ball_3D)
        print_std=0

    print_std+=1

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cap0.release()
cap1.release()
