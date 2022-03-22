import cv2
import numpy as np

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
