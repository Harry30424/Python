# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 15:50:01 2021

@author: Harry
"""

import cv2
import numpy as np
import random as rd
# import pyocr
# import pyocr.builders
# from PIL import Image

#kernel = np.ones((5,5),np.uint8) #kernel 越大越粗
# kernel1 = np.ones((5,5),np.uint8) #(5,5) 值越大越細

#read image file
#R G B
#in opencv:B G R
#G[0,255,0],B[255,0,0],R[0,0,255] fill to an image
# img = cv2.imread('lenna.jpg')
# imgContour = img.copy() #copy sourceimg
# #img = cv2.resize(img,(0,0),fx=1.0,fy=0.5)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# canny = cv2.Canny(img,150,200)
# contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# for cnt in contours:
#     # print(cut)#print shape of image
#     area = cv2.drawContours(imgContour, cnt, -1, (255,0,0),4) #-1 每個都畫 255,0,0 blue 4 粗度
#     # print(cv2.contourArea(cnt)) #shape 面積
#     # print(cv2.arcLength(cnt, True)) #True indicates shape 閉合
#     peri = cv2.arcLength(cnt,True) #edge length
#     vertices = cv2.approxPolyDP(cnt, peri*0.02, True) #近似值
#     corners = len(vertices)
#     x, y, w, h = cv2.boundingRect(vertices)
#     cv2.rectangle(imgContour, (x,y), (x+w, y+h), (0,255,0),4)#draw 方形 around shape
#     #detect shape andd putText on it.
#     if corners==3:
#         cv2.putText(imgContour, 'triangle', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
#     elif corners==4:
#         cv2.putText(imgContour, 'rectangle', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
#     elif corners==5:
#         cv2.putText(imgContour, 'pentagon', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
#     elif corners>=6:
#         cv2.putText(imgContour, 'circle', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
#detect color

# cap = cv2.VideoCapture(0)
# def empty(v):
#     pass
# cv2.namedWindow('TrackBar')
# cv2.resizeWindow('TRackBar',640,320)
# cv2.createTrackbar('Hue Min','TrackBar',0,179,empty)
# cv2.createTrackbar('Hue Max','TrackBar',179,179,empty)
# cv2.createTrackbar('Sat Min','TrackBar',0,255,empty)
# cv2.createTrackbar('Sat Max','TrackBar',255,255,empty)
# cv2.createTrackbar('Val Min','TrackBar',0,255,empty)
# cv2.createTrackbar('Val Max','TrackBar',255,255,empty)
# # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# while True:
#     h_min = cv2.getTrackbarPos('Hue Min','TrackBar')
#     h_max = cv2.getTrackbarPos('Hue Max','TrackBar')
#     s_min = cv2.getTrackbarPos('Sat Min','TrackBar')
#     s_max = cv2.getTrackbarPos('Sat Max','TrackBar')
#     v_min = cv2.getTrackbarPos('Val Min','TrackBar')
#     v_max = cv2.getTrackbarPos('Val Max','TrackBar')
#     print(h_min, h_max, s_min ,s_max, v_min ,v_max)
    
#     ret, img = cap.read()
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
#     lower = np.array([h_min, s_min, v_min])
#     upper = np.array([h_max, s_max, v_max])
#     mask = cv2.inRange(hsv,lower,upper) #bit 運算後套上mask can filted color
#     result = cv2.bitwise_and(img,img,mask=mask) 
#     cv2.imshow('img',img)
#     # cv2.imshow('hsv',hsv)
#     cv2.imshow('mask',mask)
#     cv2.imshow('result',result)
#     cv2.waitKey(1)
# print(img.shape) #imagesize
# img=np.empty((300,300,3), np.uint8) #300x300 3 is RGB
# for row in range(300):
#     for col in range(img.shape[1]):
#         img[row][col] = [rd.randint(0, 255),rd.randint(0, 255),rd.randint(0, 255)] #馬賽克
#split an image img[:150,:250]
#newImg = img[:150,:250]
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#灰階
# blur = cv2.GaussianBlur(img,(15,15),10)#模糊
# canny = cv2.Canny(img, 100, 200)#get edge of image
# dilate = cv2.dilate(canny,kernel,iterations=1) #iterations=1 粗1次
# erode = cv2.erode(dilate, kernel1,iterations=1) #erode 侵蝕

#cv2.imshow('newimg',newImg)
#cv2.imshow('gray',gray)
#cv2.imshow('blur',blur)
#cv2.imshow('canny',canny)
#cv2.imshow('dilate',dilate)
#cv2.imshow('erode',erode)


# #cv2.namedWindow('Image',0)
# img = cv2.resize(img,(500,500))

# cv2.imshow('img',img)
# cv2.waitKey(2000) #1000=1s
#cv2.destroyAllWindows()
# cv2.imshow('img',img)
# cv2.imshow('canny',canny)
# cv2.imshow('imgContour',imgContour)
# cv2.waitKey(0)
# print(type(img)) #class.numpy.ndarray






#get video
# cap = cv2.VideoCapture('Project3.mp4')
# while True:
#     ret, frame = cap.read() # ret=first, frame=second
    
#     if ret:#if get first then second
#         cv2.imshow('video',frame)
#         frame = cv2.resize(frame,(0,0),fx=0.4,fy=0.4)
#     else:
#         break        
#     if cv2.waitKey(10) == ord('q'):
#         break

#draw
#img = np.zeros((600,600,3), np.uint8)
#cv2.line(img, (0,0),(img.shape[1],img.shape[0]),(0,255,0),2)#draw a blue line
#img.shpae[0]=image width
#draw rectangle
#cv2.rectangle(img,(0,0),(400,300),(0,0,255),cv2.FILLED) #粗度2->filled 填滿
#cv2.circle(img, (300,400),30,(255,0,0), cv2.FILLED)
#cv2.putText(img, 'Hello', (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255),1)#first1 大小
#cv2.imshow('img',img)


#face detect
# img = cv2.imread('lenna.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'face_detect.xml') #load face_detect model
# faceRect = faceCascade.detectMultiScale(gray, scaleFactor=1.2 , minNeighbors=3, minSize = (32,32)) #1 img, 2倍數, 3 縮小3次
# print(len(faceRect))

# for(x, y, w, h) in faceRect:
#     cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)

# cv2.imshow('img', img)
# cv2.waitKey(0)

# def detectFace(img):
    # filename = img.split(".")[0] # 取得檔案名稱(不添加副檔名)
# img = cv2.imread('qq.jpg') # 讀取圖檔
# grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 透過轉換函式轉為灰階影像
# color = (0, 255, 0)  # 定義框的顏色

# OpenCV 人臉識別分類器
# face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 調用偵測識別人臉函式
#scaleFactor 搜索範圍的比例係數，默認值為 1.1
#minNeighbors 構成偵測目標的相鄰矩形的最小個數，默認值為 3
#minSize & maxSize 用來限制得到的目標區域範圍
# faceRects = face_classifier.detectMultiScale(
#     grayImg, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

# 大於 0 則檢測到人臉
# if len(faceRects):  
#     # 框出每一張人臉
#     for faceRect in faceRects: 
#         x, y, w, h = faceRect
#         cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
    
#     # 將結果圖片輸出
#     # cv2.imwrite(filename + "_face.jpg", img)
#     cv2.imshow('img',img)
#     cv2.waitKey(0)
# detectFace('lenna.jpg')


#virtual pen
cap = cv2.VideoCapture(0)
#blue green red
penColorHsv = [[96, 88, 139, 128 , 142, 249],
                [81, 86 , 149, 102, 208, 255],
                [0, 141, 160, 149, 255, 255]]
#pen points
penColorBGR = [[255,0,0],
                [0,255,0],
                [0,0,255]]

#[x,y,colorID] record 軌跡
drawPoints=[] #1B 2G 3R

def findPen(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for i in range(len(penColorHsv)):
        # lower = np.array([96, 88, 139])
        lower = np.array(penColorHsv[i][:3])#first 3 min
        # upper = np.array([128 , 142, 249])
        upper = np.array(penColorHsv[i][3:6])#last 3 max
        mask = cv2.inRange(hsv,lower,upper) #bit 運算後套上mask can filted color
        # result = cv2.bitwise_and(img,img,mask=mask) 
        penx, peny  = findContour(mask)
        # cv2.circle(imgContour, (penx, peny), 10, (255,0,0), cv2.FILLED)
        cv2.circle(imgContour, (penx, peny), 10, penColorBGR[i], cv2.FILLED)
        if peny!=-1:
            drawPoints.append([penx,peny,i]) #i=colorID
    # cv2.imshow('result',result)
    #turn to HSV img

def findContour(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x,y,w,h=-1,-1,-1,-1 #assignment variable
    for cnt in contours:
        # cv2.drawContours(imgContour, cnt, -1, (255,0,0),4)
        area = cv2.contourArea(cnt)
        if area>500:
            peri = cv2.arcLength(cnt, True)
            vertices = cv2.approxPolyDP(cnt, peri*0.02,True)
            x,y,w,h = cv2.boundingRect(vertices)
    return x+w//2, y

#draw graph
def draw(drawpoints):
    for point in drawpoints:
        cv2.circle(imgContour, (point[0],point[1]),10, penColorBGR[point[2]],cv2.FILLED)

while True:
    ret, frame = cap.read()
    if ret:
        imgContour = frame.copy()       #contour輪廓
        cv2.imshow('video',frame)       #main window
        findPen(frame)                  #find penColor
        draw(drawPoints)                #pen track
        cv2.imshow('countor',imgContour)#draw window
        break
    if cv2.waitKey(1) == ord('q'):
        break
# cv2.destroyAllWindows()