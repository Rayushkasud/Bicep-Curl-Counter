# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import mediapipe as mp
import numpy as np
import cv2



media_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

lCounter = 0
rCounter = 0
lPos = None
rPos = None
cap = cv2.VideoCapture(0)

#MEDIA PIPE INSTANCE

with media_pose.Pose(min_detection_confidence=0.5 , min_tracking_confidence = 0.5) as pose:
    #OBTAINING WEBCAM FEED
    while cap.isOpened():
        ret , frame = cap.read()
        width = int(cap.get(3))
        height = int(cap.get(4))
        
        cv2.rectangle(frame,(0,170),(400,210),(200,50,100),-1)
    
        img = cv2.putText(frame, 'Welcome to your Virtual Gym',(10,200),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2,cv2.LINE_AA)
    
    
        #Recolor
        image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        results = pose.process(image)
    
        #Recolor back
        image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR)
        image.flags.writeable = True

        #Extracting Landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            
            def calculateAngle(a,b,c):
                a = np.array(a)
                b = np.array(b)
                c = np.array(c)

                radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
                angle = np.abs(radians*180.0/np.pi)

                if angle > 180.0:
                    angle = 360 - angle
                return angle

            leftShoulder = landmarks[media_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[media_pose.PoseLandmark.LEFT_SHOULDER.value].y
            leftElbow = landmarks[media_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[media_pose.PoseLandmark.LEFT_ELBOW.value].y
            leftWrist = landmarks[media_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[media_pose.PoseLandmark.LEFT_WRIST.value].y
            rightShoulder = landmarks[media_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[media_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            rightElbow = landmarks[media_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[media_pose.PoseLandmark.RIGHT_ELBOW.value].y
            rightWrist = landmarks[media_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[media_pose.PoseLandmark.RIGHT_WRIST.value].y

            #calculate angles
            langle = calculateAngle(leftShoulder,leftElbow,leftWrist)
            rangle = calculateAngle(rightShoulder,rightElbow,rightWrist)
            #print(langle)

            #Render 
            cv2.putText(image,str(langle),tuple(np.multiply(leftElbow,[width,height]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(image,str(rangle),tuple(np.multiply(rightElbow,[width,height]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)

            #Count Reps
            if langle > 160:
                lPos = "down"
            if langle < 30 and lPos == "down": 
                lPos = "up"
                lCounter += 1
                print(lCounter)
                    
                        #Count Reps
            if rangle > 160:
                rPos = 'down'
            if rangle < 30 and rPos == 'down':

                rPos = 'up'
                rCounter += 1
                print(rCounter)

        except:
            pass

        #Render rep counter
        cv2.rectangle(image,(180,0),(400,120),(200,50,100),-1)
        cv2.putText(image,'REPS:',(200,30),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(image,'left-',(200,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(image,str(lCounter),(200,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        
        cv2.rectangle(image,(830,0),(1050,120),(200,50,100),-1)
        cv2.putText(image,'REPS:',(850,30),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(image,'right-',(850,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(image,str(rCounter),(850,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
    
        #Render detections
        mp_drawing.draw_landmarks(image,results.pose_landmarks ,media_pose.POSE_CONNECTIONS ,
                                mp_drawing.DrawingSpec(color=(50,50,199),thickness=2,circle_radius=2),
                                mp_drawing.DrawingSpec(color=(150,30,100), thickness=2,circle_radius=2))

        
    
        cv2.imshow('frame',image)
        

    
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break




cap.release()
cv2.destroyAllWindows()
    