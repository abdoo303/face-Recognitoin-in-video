import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def rescale(frame, scale=.75):
    width= int(frame.shape[1]*scale)
    height= int(frame.shape[0]*scale)
    dimensions= (height, width)
    return  cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

def display_img(path):
    image= cv.imread(path)
    detect(image)
    cv.imshow('image',image)
    cv.waitKey(0)

def detect(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    haarcasc = cv.CascadeClassifier(r'face.xml')
    face_rect = haarcasc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9)
    for (x,y,w,h) in face_rect:
        rectangle = cv.rectangle(img, (x,y), (x+w,y+h), (0, 0, 255), 2)


def display_vid(path=0):
    capture = cv.VideoCapture(path)
    while True:
        isTrue,frame= capture.read()
        frame_scaled=cv.resize(frame,(1000,1000))
        #cv.putText(frame_scaled,"that's Abdelrahim",(25,25),cv.FONT_HERSHEY_TRIPLEX,1,(0,255,0),2)
        detect(frame_scaled)
        cv.imshow("frm",frame_scaled)
        if cv.waitKey(20)& 0xFF==ord('q'):
            break
    capture.release()
    cv.destroyAllWindows()

frame=cv.imread(r"C:\Users\DELL\Pictures\Saved Pictures\photo_2022-05-25_08-36-53.jpg")

blur= cv.GaussianBlur(frame.copy(),(5,5),cv.BORDER_DEFAULT)
canny= cv.Canny(frame,125,175)
hsv= cv.cvtColor(frame.copy(),cv.COLOR_BGR2HSV)
gray=cv.cvtColor(frame.copy(),cv.COLOR_BGR2GRAY)
lab=cv.cvtColor(frame.copy(),cv.COLOR_BGR2LAB)
rgb=cv.cvtColor(frame.copy(),cv.COLOR_BGR2RGB)
b,g,r= cv.split(frame.copy())
merged= cv.merge([g,b,r])
#contours, hier= cv.findContours(frame,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
blank= np.zeros((400,400),dtype='uint8')
rectangle= cv.rectangle(blank.copy(),(20,20),(380,380),255,-1)
circle= cv.circle(blank.copy(),(200,200),200,255,-1)
'''
cv.imshow("REC",rectangle
)
cv.imshow('CIR',circle)
colors=['b','g','r']
plt.figure()
plt.xlabel("bins")
plt.ylabel('density')
for i,c in enumerate(colors):
    hist= cv.calcHist([frame],[i],None,[256],[0,256])
    plt.plot(hist,color=c)
    plt.show()

threshold,thresh=cv.threshold(frame.copy(),200,255,cv.THRESH_BINARY)
thresh_adap=cv.adaptiveThreshold(frame.copy(),255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,3)

cv.imshow('binarized',thresh_adap)
cv.waitKey(0)
thresh= cv.cvtColor(thresh,cv.COLOR_BGR2RGB)
'''
#adisplay_img(r'C:\Users\DELL\Pictures\Camera Roll\WIN_20220218_14_22_32_Pro.jpg')
display_vid()
#display_img(r"C:\Users\DELL\Pictures\Saved Pictures\pictures.jpg")

