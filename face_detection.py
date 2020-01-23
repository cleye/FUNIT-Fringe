import numpy as np
import cv2
from imutils.video import VideoStream
from imutils import resize
import time



vs = VideoStream(src=0).start()
print('Warming up')
time.sleep(2.0)
print('Program started')

cv2.namedWindow("Output")


face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile('haarcascade_frontalface_alt.xml')):
    print('--(!)Error loading face cascade')
    exit(0)


while True:
    # preparation(vs,prep_time,rotate)
    frame = vs.read()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    #-- Detect multiple faces
    faces = face_cascade.detectMultiScale(frame_gray)

    for (x,y,w,h) in faces:
        # use 1.25x the detected rectangle
        x1 = int(x-0.25*w)
        y1 = int(y-0.25*h)

        x2 = int(x+1.25*w)
        y2 = int(y+1.25*h)

        # draw rect over face
        # frame = cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 4)

        # crop frame
        frame = frame[y1:y2, x1:x2]

        # faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        # eyes = eyes_cascade.detectMultiScale(faceROI)
        # for (x2,y2,w2,h2) in eyes:
        #     eye_center = (x + x2 + w2//2, y + y2 + h2//2)
        #     radius = int(round((w2 + h2)*0.25))
        #     frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)



    # Inference
    cv2.imshow('Output', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break




