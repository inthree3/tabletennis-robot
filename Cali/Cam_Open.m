webcamlist
cam1 = webcam(1);
cam2 = webcam(2);

preview(cam1);
preview(cam2);

cam1.AvailableResolutions;
cam2.AvailableResolutions;

cam1.Resolution='1280x720';
cam2.Resolution='1280x720';

img = snapshot(cam1);
imshow(img);