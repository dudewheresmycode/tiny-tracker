# import the necessary packages
from collections import deque
import cv2
import numpy as np
import imutils
import argparse
from configparser import ConfigParser
from ColorModuleExtended import ColorFinder
import math
import time
import sys
import os
import ast
import json

from ball_colors import *
from ball_utils import *

# Startpoint Zone
ballradius = 0
darkness = 0
flipImage = 0
mjpegenabled = 0
ps4=0
overwriteFPS = 0
golfballradius = 21.33; # in mm
customhsv = {}
timeSinceTriggered = 0
noOfStarts = 0
frameskip = 0

textcolor=(200, 200, 200)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-i", "--img",
                help="path to the (optional) image file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size - default is 64")
ap.add_argument("-w", "--camera", type=int, default=0,
                help="webcam index number - default is 0")
ap.add_argument("-c", "--ballcolor",
                help="ball color - default is yellow")
ap.add_argument("-d", "--debug",
                help="debug - color finder and wait timer")
ap.add_argument("-r", "--resize", type=int, default=640,
                help="window resize in width pixel - default is 640px")
ap.add_argument("-g", "--config",
                help="Path to config.ini file")
args = vars(ap.parse_args())

# Parse config
parser = ConfigParser()
configFilePath = 'config.ini'
if args.get("config", False):
    configFilePath = args['config']

debugLog("Loading config file: "+configFilePath)
parser.read(configFilePath)

if parser.has_option('putting', 'startx1'):
    sx1=int(parser.get('putting', 'startx1'))
else:
    sx1=10
if parser.has_option('putting', 'startx2'):
    sx2=int(parser.get('putting', 'startx2'))
else:
    sx2=180
if parser.has_option('putting', 'y1'):
    y1=int(parser.get('putting', 'y1'))
else:
    y1=180
if parser.has_option('putting', 'y2'):
    y2=int(parser.get('putting', 'y2'))
else:
    y2=450
if parser.has_option('putting', 'radius'):
    ballradius=int(parser.get('putting', 'radius'))
else:
    ballradius=0
if parser.has_option('putting', 'flip'):
    flipImage=int(parser.get('putting', 'flip'))
else:
    flipImage=0
if parser.has_option('putting', 'flipview'):
    flipView=int(parser.get('putting', 'flipview'))
else:
    flipView=0
if parser.has_option('putting', 'darkness'):
    darkness=int(parser.get('putting', 'darkness'))
else:
    darkness=0
if parser.has_option('putting', 'mjpeg'):
    mjpegenabled=int(parser.get('putting', 'mjpeg'))
else:
    mjpegenabled=0
if parser.has_option('putting', 'ps4'):
    ps4=int(parser.get('putting', 'ps4'))
else:
    ps4=0
if parser.has_option('putting', 'fps'):
    overwriteFPS=int(parser.get('putting', 'fps'))
else:
    overwriteFPS=0
if parser.has_option('putting', 'height'):
    height=int(parser.get('putting', 'height'))
else:
    height=360
if parser.has_option('putting', 'width'):
    width=int(parser.get('putting', 'width'))
else:
    width=640
if parser.has_option('putting', 'customhsv'):
    customhsv=ast.literal_eval(parser.get('putting', 'customhsv'))
    debugLog(customhsv)
else:
    customhsv={}


## Init variables

# Globals

# Detection Gateway
x1=sx2+10
x2=x1+10

#coord of polygon in frame::: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
startcoord=[[sx1,y1],[sx2,y1],[sx1,y2],[sx2,y2]]

#coord of polygon in frame::: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
coord=[[x1,y1],[x2,y1],[x1,y2],[x2,y2]]


actualFPS = 0

videoStartTime = time.time()

# initialize variables to store the start and end positions of the ball
startCircle = (0, 0, 0)
endCircle = (0, 0, 0)
startPos = (0,0)
endPos = (0,0)
startTime = time.time()
timeSinceEntered = 0
replaytimeSinceEntered = 0
pixelmmratio = 0

# initialize variable to store start candidates of balls
startCandidates = []
startminimum = 30

# Initialize Entered indicator
entered = False
started = False
left = False

lastShotStart = (0,0)
lastShotEnd = (0,0)
lastShotSpeed = 0
lastShotHLA = 0 

speed = 0

tim1 = 0
tim2 = 0
replaytrigger = 0

# calibration

colorcount = 0
calibrationtime = time.time()
calObjectCount = 0
calColorObjectCount = []
calibrationTimeFrame = 30

# Calibrate Recording Indicator

record = True

# Videofile Indicator

videofile = False

# remove duplicate advanced screens for multipla 'a' and 'd' key presses)
a_key_pressed = False 
d_key_pressed = False 


# defaults to the orange option
hsvVals = orange

if customhsv == {}:
    if args.get("ballcolor", False):
        if args["ballcolor"] == "white":
            hsvVals = white
        elif args["ballcolor"] == "white2":
            hsvVals = white2
        elif args["ballcolor"] ==  "yellow":
            hsvVals = yellow 
        elif args["ballcolor"] ==  "yellow2":
            hsvVals = yellow2 
        elif args["ballcolor"] ==  "orange":
            hsvVals = orange
        elif args["ballcolor"] ==  "orange2":
            hsvVals = orange2
        elif args["ballcolor"] ==  "orange3":
            hsvVals = orange3
        elif args["ballcolor"] ==  "orange4":
            hsvVals = orange4
        elif args["ballcolor"] ==  "green":
            hsvVals = green 
        elif args["ballcolor"] ==  "green2":
            hsvVals = green2               
        elif args["ballcolor"] ==  "red":
            hsvVals = red             
        elif args["ballcolor"] ==  "red2":
            hsvVals = red2             
        else:
            hsvVals = yellow

        if args["ballcolor"] is not None:
            debugLog("Ballcolor: "+str(args["ballcolor"]))
else:
    hsvVals = customhsv
    debugLog("Custom HSV Values set in config.ini")

def resizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def drawAlphaRect(image):
    overlay = image.copy()
    x, y, w, h = 0, 0, 640, 90  # Rectangle parameters
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 0), -1)  # A filled rectangle
    alpha = 0.4  # Transparency factor.
    # Following line overlays transparent rectangle over the image
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
# 
# Start Splash Screen
splashImagePath = resource_path("images/splash.png")
debugLog(splashImagePath)
frame = cv2.imread(splashImagePath)
origframe2 = cv2.imread(splashImagePath)
cv2.putText(frame, "Starting Video: Try MJPEG option in advanced settings for faster startup",(20,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,textcolor)
outputframe = resizeWithAspectRatio(frame, width=int(args["resize"]))
cv2.imshow("Putting View: Press q to exit / a for adv. settings", outputframe)


# Create the color Finder object set to True if you need to Find the color

if args.get("debug", False):
    myColorFinder = ColorFinder(True)
    myColorFinder.setTrackbarValues(hsvVals)
else:
    myColorFinder = ColorFinder(False)

### Setup camera

pts = deque(maxlen=args["buffer"])
tims = deque(maxlen=args["buffer"])
fpsqueue = deque(maxlen=240)
replay1queue = deque(maxlen=600)
replay2queue = deque(maxlen=600)

webcamindex = 0

message = ""


# if a webcam index is supplied, grab the reference
if args.get("camera", False):
    webcamindex = args["camera"]
    debugLog("Putting Cam activated at "+str(webcamindex))

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    if mjpegenabled == 0:
        vs = cv2.VideoCapture(webcamindex)
    else:
        vs = cv2.VideoCapture(webcamindex + cv2.CAP_DSHOW)
        # Check if FPS is overwritten in config
        if overwriteFPS != 0:
            vs.set(cv2.CAP_PROP_FPS, overwriteFPS)
            debugLog("Overwrite FPS: "+str(vs.get(cv2.CAP_PROP_FPS)))
        if height != 0 and width != 0:
            vs.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            vs.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        mjpeg = cv2.VideoWriter_fourcc('M','J','P','G')
        vs.set(cv2.CAP_PROP_FOURCC, mjpeg)
    if vs.get(cv2.CAP_PROP_BACKEND) == -1:
        message = "No Camera could be opened at webcamera index "+str(webcamindex)+". If your webcam only supports compressed format MJPEG instead of YUY2 please set MJPEG option to 1"
    else:
        if ps4 == 1:
            vs.set(cv2.CAP_PROP_FPS, 120)
            vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1724)
            vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 404)
            #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3448)
            #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 808)
        debugLog("Backend: "+str(vs.get(cv2.CAP_PROP_BACKEND)))
        debugLog("FourCC: "+str(vs.get(cv2.CAP_PROP_FOURCC)))
        debugLog("FPS: "+str(vs.get(cv2.CAP_PROP_FPS)))
else:
    vs = cv2.VideoCapture(args["video"])
    videofile = True

# Get video metadata

video_fps = vs.get(cv2.CAP_PROP_FPS)
height = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = vs.get(cv2.CAP_PROP_FRAME_WIDTH)

if parser.has_option('putting', 'saturation'):
    saturation=float(parser.get('putting', 'saturation'))
else:
    saturation = vs.get(cv2.CAP_PROP_SATURATION)
if parser.has_option('putting', 'exposure'):
    exposure=float(parser.get('putting', 'exposure'))
else:
    exposure = vs.get(cv2.CAP_PROP_EXPOSURE)
if parser.has_option('putting', 'autowb'):
    autowb=float(parser.get('putting', 'autowb'))
else:
    autowb = vs.get(cv2.CAP_PROP_AUTO_WB)
if parser.has_option('putting', 'whiteBalanceBlue'):
    whiteBalanceBlue=float(parser.get('putting', 'whiteBalanceBlue'))
else:
    whiteBalanceBlue = vs.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U)
if parser.has_option('putting', 'whiteBalanceRed'):
    whiteBalanceRed=float(parser.get('putting', 'whiteBalanceRed'))
else:
    whiteBalanceRed = vs.get(cv2.CAP_PROP_WHITE_BALANCE_RED_V)
if parser.has_option('putting', 'brightness'):
    brightness=float(parser.get('putting', 'brightness'))
else:
    brightness = vs.get(cv2.CAP_PROP_BRIGHTNESS)
if parser.has_option('putting', 'contrast'):
    contrast=float(parser.get('putting', 'contrast'))
else:
    contrast = vs.get(cv2.CAP_PROP_CONTRAST)
if parser.has_option('putting', 'hue'):
    hue=float(parser.get('putting', 'hue'))
else:
    hue = vs.get(cv2.CAP_PROP_HUE)
if parser.has_option('putting', 'gain'):
    gain=float(parser.get('putting', 'gain'))
else:
    gain = vs.get(cv2.CAP_PROP_HUE)
if parser.has_option('putting', 'monochrome'):
    monochrome=float(parser.get('putting', 'monochrome'))
else:
    monochrome = vs.get(cv2.CAP_PROP_MONOCHROME)
if parser.has_option('putting', 'sharpness'):
    sharpness=float(parser.get('putting', 'sharpness'))
else:
    sharpness = vs.get(cv2.CAP_PROP_SHARPNESS)
if parser.has_option('putting', 'autoexposure'):
    autoexposure=float(parser.get('putting', 'autoexposure'))
else:
    autoexposure = vs.get(cv2.CAP_PROP_AUTO_EXPOSURE)
if parser.has_option('putting', 'gamma'):
    gamma=float(parser.get('putting', 'gamma'))
else:
    gamma = vs.get(cv2.CAP_PROP_GAMMA)
if parser.has_option('putting', 'zoom'):
    zoom=float(parser.get('putting', 'zoom'))
else:
    zoom = vs.get(cv2.CAP_PROP_ZOOM)
    gamma = vs.get(cv2.CAP_PROP_GAMMA)
if parser.has_option('putting', 'focus'):
    focus=float(parser.get('putting', 'focus'))
else:
    focus = vs.get(cv2.CAP_PROP_FOCUS)
if parser.has_option('putting', 'autofocus'):
    autofocus=float(parser.get('putting', 'autofocus'))
else:
    autofocus = vs.get(cv2.CAP_PROP_AUTOFOCUS)

vs.set(cv2.CAP_PROP_SATURATION,saturation)
vs.set(cv2.CAP_PROP_EXPOSURE,exposure)
vs.set(cv2.CAP_PROP_AUTO_WB,autowb)
vs.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U,whiteBalanceBlue)
vs.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V,whiteBalanceRed)
vs.set(cv2.CAP_PROP_BRIGHTNESS,brightness)
vs.set(cv2.CAP_PROP_CONTRAST,contrast)
vs.set(cv2.CAP_PROP_HUE,hue)
vs.set(cv2.CAP_PROP_GAIN,gain)
vs.set(cv2.CAP_PROP_MONOCHROME,monochrome)
vs.set(cv2.CAP_PROP_SHARPNESS,sharpness)
vs.set(cv2.CAP_PROP_AUTO_EXPOSURE,autoexposure)
vs.set(cv2.CAP_PROP_GAMMA,gamma)
vs.set(cv2.CAP_PROP_ZOOM,zoom)
vs.set(cv2.CAP_PROP_FOCUS,focus)
vs.set(cv2.CAP_PROP_AUTOFOCUS,autofocus)


debugLog("video_fps: "+str(video_fps))
debugLog("height: "+str(height))
debugLog("width: "+str(width))

if type(video_fps) == float:
    if video_fps == 0.0:
        e = vs.set(cv2.CAP_PROP_FPS, 60)
        new_fps = []
        new_fps.append(0)

    if video_fps > 0.0:
        new_fps = []
        new_fps.append(video_fps)
    video_fps = new_fps

def decode(myframe):
    left = np.zeros((400,632,3), np.uint8)
    right = np.zeros((400,632,3), np.uint8)
    
    for i in range(400):
        left[i] = myframe[i, 32: 640 + 24] 
        right[i] = myframe[i, 640 + 24: 640 + 24 + 632] 
    
    return (left, right)

def setFPS(value):
    debugLog(value)
    vs.set(cv2.CAP_PROP_FPS,value)
    pass 

def setXStart(value):
    debugLog(value)
    startcoord[0][0]=value
    startcoord[2][0]=value

    global sx1
    sx1=int(value)    
    parser.set('putting', 'startx1', str(sx1))
    parser.write(open(CFG_FILE, "w"))
    pass

def setXEnd(value):
    debugLog(value)
    startcoord[1][0]=value
    startcoord[3][0]=value 

    global x1
    global x2
    global sx2
     
    # Detection Gateway
    x1=int(value+10)
    x2=int(x1+10)

    #coord=[[x1,y1],[x2,y1],[x1,y2],[x2,y2]]
    coord[0][0]=x1
    coord[2][0]=x1
    coord[1][0]=x2
    coord[3][0]=x2

    sx2=int(value)    
    parser.set('putting', 'startx2', str(sx2))
    parser.write(open(CFG_FILE, "w"))
    pass  

def setYStart(value):
    debugLog(value)
    startcoord[0][1]=value
    startcoord[1][1]=value

    global y1

    #coord=[[x1,y1],[x2,y1],[x1,y2],[x2,y2]]
    coord[0][1]=value   
    coord[1][1]=value

    y1=int(value)    
    parser.set('putting', 'y1', str(y1))
    parser.write(open(CFG_FILE, "w"))     
    pass

def setYEnd(value):
    debugLog(value)
    startcoord[2][1]=value
    startcoord[3][1]=value 

    global y2

    #coord=[[x1,y1],[x2,y1],[x1,y2],[x2,y2]]
    coord[2][1]=value   
    coord[3][1]=value

    y2=int(value)    
    parser.set('putting', 'y2', str(y2))
    parser.write(open(CFG_FILE, "w"))     
    pass 

def setBallRadius(value):
    debugLog(value)    
    global ballradius
    ballradius = int(value)
    parser.set('putting', 'radius', str(ballradius))
    parser.write(open(CFG_FILE, "w"))
    pass

def setFlip(value):
    debugLog(value)    
    global flipImage
    flipImage = int(value)
    parser.set('putting', 'flip', str(flipImage))
    parser.write(open(CFG_FILE, "w"))
    pass

def setFlipView(value):
    debugLog(value)    
    global flipView
    flipView = int(value)
    parser.set('putting', 'flipView', str(flipView))
    parser.write(open(CFG_FILE, "w"))
    pass

def setMjpeg(value):
    debugLog(value)    
    global mjpegenabled
    global message
    if mjpegenabled != int(value):
        vs.release()
        message = "Video Codec changed - Please restart the putting app"
    mjpegenabled = int(value)
    parser.set('putting', 'mjpeg', str(mjpegenabled))
    parser.write(open(CFG_FILE, "w"))
    pass

def setOverwriteFPS(value):
    debugLog(value)    
    global overwriteFPS
    global message
    if overwriteFPS != int(value):
        vs.release()
        message = "Overwrite of FPS changed - Please restart the putting app"
    overwriteFPS = int(value)
    parser.set('putting', 'fps', str(overwriteFPS))
    parser.write(open(CFG_FILE, "w"))
    pass

def setDarkness(value):
    debugLog(value)    
    global darkness
    darkness = int(value)
    parser.set('putting', 'darkness', str(darkness))
    parser.write(open(CFG_FILE, "w"))
    pass

def GetAngle (p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dX = x2 - x1
    dY = y2 - y1
    rads = math.atan2 (-dY, dX)

    if flipImage == 1 and videofile == False:    	
        rads = rads*-1
    return math.degrees (rads)

def rgb2yuv(rgb):
    m = np.array([
        [0.29900, -0.147108,  0.614777],
        [0.58700, -0.288804, -0.514799],
        [0.11400,  0.435912, -0.099978]
    ])
    yuv = np.dot(rgb, m)
    yuv[:,:,1:] += 0.5
    return yuv

def yuv2rgb(yuv):
    m = np.array([
        [1.000,  1.000, 1.000],
        [0.000, -0.394, 2.032],
        [1.140, -0.581, 0.000],
    ])
    yuv[:, :, 1:] -= 0.5
    rgb = np.dot(yuv, m)
    return rgb


# allow the camera or video file to warm up
time.sleep(0.5)

previousFrame = cv2.Mat

while True:
    # set the frameTime
    frameTime = time.time()
    fpsqueue.append(frameTime)
    
    actualFPS = actualFPS + 1
    videoTimeDiff = fpsqueue[len(fpsqueue)-1] - fpsqueue[0]
    if videoTimeDiff != 0:
        fps = len(fpsqueue) / videoTimeDiff
    else:
        fps = 0

    # get webcam frame
    ret, frame = vs.read()
    if ps4 == 1 and ret == True:
        leftframe, rightframe = decode(frame)
        frame = leftframe[0:400,20:632]
        width = 612
        height = 400
    # flip image on y-axis
    if flipImage == 1 and videofile == False:	
        frame = cv2.flip(frame, flipImage)


    # handle the frame from VideoCapture or VideoStream
    # frame = frame[1] if args.get("video", False) else frame

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        debugLog("no frame")
        frame = cv2.imread(splashImagePath)
        cv2.putText(frame,"Error: "+"No Frame",(20, 20),cv2.FONT_HERSHEY_SIMPLEX,0.5,textcolor)
        cv2.putText(frame,"Message: "+message,(20, 40),cv2.FONT_HERSHEY_SIMPLEX,0.5,textcolor)
        cv2.imshow("Putting View: Press q to exit / a for adv. settings", frame)
        cv2.waitKey(0)
        break

    origframe = frame.copy()
    cv2.normalize(frame, frame, 0-darkness, 255-darkness, norm_type=cv2.NORM_MINMAX)
    
    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=640, height=360)
    #origframe2 = imutils.resize(origframe2, width=640, height=360) 
    #origframe = imutils.resize(frame, width=640, height=360)  
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Find the Color Ball
    imgColor, mask, newHSV = myColorFinder.update(hsv, hsvVals)
    if hsvVals != newHSV:
        debugLog(newHSV)
        parser.set('putting', 'customhsv', str(newHSV)) #['hmin']+newHSV['smin']+newHSV['vmin']+newHSV['hmax']+newHSV['smax']+newHSV['vmax']))
        parser.write(open(CFG_FILE, "w"))
        hsvVals = newHSV
        debugLog("HSV values changed - Custom Color Set to config.ini")

    mask = mask[y1:y2, sx1:640]
    
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    
    # Startpoint Zone
    cv2.line(frame, (startcoord[0][0], startcoord[0][1]), (startcoord[1][0], startcoord[1][1]), (0, 210, 255), 2)  # First horizontal line
    cv2.line(frame, (startcoord[0][0], startcoord[0][1]), (startcoord[2][0], startcoord[2][1]), (0, 210, 255), 2)  # Vertical left line
    cv2.line(frame, (startcoord[2][0], startcoord[2][1]), (startcoord[3][0], startcoord[3][1]), (0, 210, 255), 2)  # Second horizontal line
    cv2.line(frame, (startcoord[1][0], startcoord[1][1]), (startcoord[3][0], startcoord[3][1]), (0, 210, 255), 2)  # Vertical right line

    # Detection Gateway
    cv2.line(frame, (coord[0][0], coord[0][1]), (coord[1][0], coord[1][1]), (0, 0, 255), 2)  # First horizontal line
    cv2.line(frame, (coord[0][0], coord[0][1]), (coord[2][0], coord[2][1]), (0, 0, 255), 2)  # Vertical left line
    cv2.line(frame, (coord[2][0], coord[2][1]), (coord[3][0], coord[3][1]), (0, 0, 255), 2)  # Second horizontal line
    cv2.line(frame, (coord[1][0], coord[1][1]), (coord[3][0], coord[3][1]), (0, 0, 255), 2)  # Vertical right line


    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # only proceed if at least one contour was found
    if len(cnts) > 0:

        x = 0
        y = 0
        radius = 0
        center= (0,0)
        
        for index in range(len(cnts)):
            circle = (0,0,0)
            center= (0,0)
            radius = 0
            # Eliminate countours that are outside the y dimensions of the detection zone
            ((tempcenterx, tempcentery), tempradius) = cv2.minEnclosingCircle(cnts[index])
            tempcenterx = tempcenterx + sx1
            tempcentery = tempcentery + y1
            if (tempcentery >= y1 and tempcentery <= y2):
                rangefactor = 50
                cv2.drawContours(mask, cnts, index, (60, 255, 255), 1)
                #cv2.putText(frame,"Radius:"+str(int(tempradius)),(int(tempcenterx)+3, int(tempcentery)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))
                # Eliminate countours significantly different than startCircle by comparing radius in range
                if (started == True and startCircle[2]+rangefactor > tempradius and startCircle[2]-rangefactor < tempradius):
                    x = int(tempcenterx)
                    y = int(tempcentery)
                    radius = int(tempradius)
                    center= (x,y)
                else:
                    if not started:
                        x = int(tempcenterx)
                        y = int(tempcentery)
                        radius = int(tempradius)
                        center= (x,y)
                        #debugLog("No Startpoint Set Yet: "+str(center)+" "+str(startCircle[2]+rangefactor)+" > "+str(radius)+" AND "+str(startCircle[2]-rangefactor)+" < "+str(radius))
            else:
                break

            # only proceed if the radius meets a minimum size
            if radius >=5:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points  
                circle = (x,y,radius)
                if circle:
                    # check if the circle is stable to detect if a new start is there
                    if not started or startPos[0]+10 <= center[0] or startPos[0]-10 >= center[0]:
                        if (center[0] >= sx1 and center[0] <= sx2):
                            startCandidates.append(center)
                            if len(startCandidates) > startminimum :
                                startCandidates.pop(0)
                                arr = np.array(startCandidates)
                                # Create an empty list
                                filter_arr = []
                                # go through each element in arr
                                for element in arr:
                                # if the element is completely divisble by 2, set the value to True, otherwise False
                                    if (element[0] == center[0] and center[1] == element[1]):
                                        filter_arr.append(True)
                                    else:
                                        filter_arr.append(False)

                                filtered = arr[filter_arr]

                                if len(filtered) >= (startminimum/2):
                                    debugLog("New ball start position found!")
                                    sendEvent({ "eventName": "ready", "data": { "position": center } })
                                    # replayavail = False
                                    noOfStarts = noOfStarts + 1
                                    lastShotSpeed = 0
                                    pts.clear()
                                    tims.clear()
                                    filteredcircles = []
                                    filteredcircles.append(circle)
                                    startCircle = circle
                                    startPos = center
                                    startTime = frameTime
                                    #debugLog("Start Position: "+ str(startPos[0]) +":" + str(startPos[1]))
                                    # Calculate the pixel per mm ratio according to z value of circle and standard radius of 2133 mm
                                    if ballradius == 0:
                                        pixelmmratio = circle[2] / golfballradius
                                    else:
                                        pixelmmratio = ballradius / golfballradius

                                    started = True
                                    # replay = True
                                    # replaytrigger = 0          
                                    entered = False
                                    left = False
                                    # update the points and tims queues
                                    pts.appendleft(center)
                                    tims.appendleft(frameTime)  
                                    # global replay1
                                    # global replay2

                                    # replay1 = cv2.VideoWriter('replay1/Replay1_'+ str(noOfStarts) +'.mp4', apiPreference=0, fourcc=fourcc,fps=120, frameSize=(int(width), int(height)))
                                    # if replaycam == 1:
                                    #     replay2 = cv2.VideoWriter('replay2/Replay2_'+ str(noOfStarts) +'.mp4', apiPreference=0, fourcc=fourcc,fps=120, frameSize=(int(replaycamwidth), int(replaycamheight)))

                        else:

                            if (x >= coord[0][0] and entered == False and started == True):
                                cv2.line(frame, (coord[0][0], coord[0][1]), (coord[2][0], coord[2][1]), (0, 255, 0),2)  # Changes line color to green
                                tim1 = frameTime
                                debugLog("Ball Entered. Position: "+str(center))
                                startPos = center
                                entered = True
                                # update the points and tims queues
                                pts.appendleft(center)
                                tims.appendleft(frameTime)
                                
                                break
                            else:
                                if ( x > coord[1][0] and entered == True and started == True):
                                    #calculate hla for circle and pts[0]
                                    previousHLA = (GetAngle((startCircle[0],startCircle[1]),pts[0])*-1)
                                    #calculate hla for circle and now
                                    currentHLA = (GetAngle((startCircle[0],startCircle[1]),center)*-1)
                                    #check if HLA is inverted
                                    similarHLA = False
                                    if left == True:
                                        if ((previousHLA <= 0 and currentHLA <=2) or (previousHLA >= 0 and currentHLA >=-2)):
                                            hldDiff = (pow(currentHLA, 2) - pow(previousHLA, 2))
                                            if  hldDiff < 30:
                                                similarHLA = True
                                        else:
                                            similarHLA = False
                                    else:
                                        similarHLA = True
                                    if ( x > (pts[0][0]+50)and similarHLA == True): # and (pow((y - (pts[0][1])), 2)) <= pow((y - (pts[1][1])), 2) 
                                        cv2.line(frame, (coord[1][0], coord[1][1]), (coord[3][0], coord[3][1]), (0, 255, 0),2)  # Changes line color to green
                                        tim2 = frameTime # Final time
                                        debugLog("Ball Left. Position: "+str(center))
                                        left = True
                                        endPos = center
                                        # calculate the distance traveled by the ball in pixel
                                        a = endPos[0] - startPos[0]
                                        b = endPos[1] - startPos[1]
                                        distanceTraveled = math.sqrt( a*a + b*b )
                                        if not pixelmmratio is None:
                                            # convert the distance traveled to mm using the pixel ratio
                                            distanceTraveledMM = distanceTraveled / pixelmmratio
                                            # take the time diff from ball entered to this frame
                                            timeElapsedSeconds = (tim2 - tim1)
                                            # calculate the speed in MPH
                                            if not timeElapsedSeconds  == 0:
                                                speed = ((distanceTraveledMM / 1000 / 1000) / (timeElapsedSeconds)) * 60 * 60 * 0.621371
                                            # debug out
                                            debugLog("Time Elapsed in Sec: "+str(timeElapsedSeconds))
                                            debugLog("Distance travelled in MM: "+str(distanceTraveledMM))
                                            debugLog("Speed: "+str(speed)+" MPH")
                                            # update the points and tims queues
                                            pts.appendleft(center)
                                            tims.appendleft(frameTime)
                                            break
                                    else:
                                        debugLog("False Exit after the Ball")

                                        # flip image on y-axis for view only





    # loop over the set of tracked points
    if len(pts) != 0 and entered == True:
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 1.5)


        timeSinceEntered = (frameTime - tim1)
        replaytrigger = tim1

    if left == True:

        # Send Shot Data
        if (tim2 and timeSinceEntered > 0.5 and distanceTraveledMM and timeElapsedSeconds and speed >= 0.5 and speed <= 25):
            debugLog("----- Shot Complete --------")
            debugLog("Time Elapsed in Sec: "+str(timeElapsedSeconds))
            debugLog("Distance travelled in MM: "+str(distanceTraveledMM))
            debugLog("Speed: "+str(speed)+" MPH")

            #     ballSpeed: ballData.BallSpeed,
            #     totalSpin: ballData.TotalSpin,
            totalSpin = 0
            #     hla: ballData.LaunchDirection,
            launchDirection = (GetAngle((startCircle[0],startCircle[1]),endPos)*-1)
            debugLog("HLA: Line"+str((startCircle[0],startCircle[1]))+" Angle "+str(launchDirection))
            #Decimal(launchDirection);
            if (launchDirection > -40 and launchDirection < 40):

                lastShotStart = (startCircle[0],startCircle[1])
                lastShotEnd = endPos
                lastShotSpeed = speed
                lastShotHLA = launchDirection
                    
                # Data that we will send in post request.
                data = {"ballSpeed":"%.2f" % speed,"totalSpin":totalSpin,"launchDirection":"%.2f" % launchDirection}

                # The POST request to our node server
                if args["ballcolor"] == "calibrate":
                    debugLog("calibration mode - shot data not send")
                else:
                    sendEvent({ "eventName": "putt", "data": data })
                    # TODO: allow http requests as an optional command line argument?
                    # try:
                    #     res = requests.post('http://127.0.0.1:8888/putting', json=data)
                    #     res.raise_for_status()
                    #     # Convert response data to json
                    #     returned_data = res.json()

                    #     debugLog(returned_data)
                    #     result = returned_data['result']
                    #     debugLog("Response from Node.js:", result)

                    # except requests.exceptions.HTTPError as e:  # This is the correct syntax
                    #     debugLog(e)
                    # except requests.exceptions.RequestException as e:  # This is the correct syntax
                    #     debugLog(e)
            else:
                debugLog("Misread on HLA - Shot not send!!!")    
            if len(pts) > calObjectCount:
                calObjectCount = len(pts)
            debugLog("----- Data reset --------")
            started = False
            entered = False
            left = False
            speed = 0
            timeSinceEntered = 0
            tim1 = 0
            tim2 = 0
            distanceTraveledMM = 0
            timeElapsedSeconds = 0
            startCircle = (0, 0, 0)
            endCircle = (0, 0, 0)
            startPos = (0,0)
            endPos = (0,0)
            startTime = time.time()
            pixelmmratio = 0
            pts.clear()
            tims.clear()

            # Further clearing - startPos, endPos
    else:
        # Send Shot Data
        if (tim1 and timeSinceEntered > 0.5):
            debugLog("----- Data reset --------")
            started = False
            entered = False
            left = False
            replay = False
            speed = 0
            timeSinceEntered = 0
            tim1 = 0
            tim2 = 0
            distanceTraveledMM = 0
            timeElapsedSeconds = 0
            startCircle = (0, 0, 0)
            endCircle = (0, 0, 0)
            startPos = (0,0)
            endPos = (0,0)
            startTime = time.time()
            pixelmmratio = 0
            pts.clear()
            tims.clear()
            
    #cv2.putText(frame,"entered:"+str(entered),(20,180),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))
    #cv2.putText(frame,"FPS:"+str(fps),(20,200),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))

    if not lastShotSpeed == 0:
        cv2.line(frame,(lastShotStart),(lastShotEnd),(0, 255, 255),4,cv2.LINE_AA)      
    
    if started:
        cv2.line(frame,(sx2,startCircle[1]),(sx2+400,startCircle[1]),(255, 255, 255),4,cv2.LINE_AA)
    else:
        cv2.line(frame,(sx2,int(y1+((y2-y1)/2))),(sx2+400,int(y1+((y2-y1)/2))),(255, 255, 255),4,cv2.LINE_AA) 

        # Mark Start Circle
    if started:
        cv2.circle(frame, (startCircle[0],startCircle[1]), startCircle[2],(0, 0, 255), 2)
        cv2.circle(frame, (startCircle[0],startCircle[1]), 5, (0, 0, 255), -1) 

    # Mark Entered Circle
    if entered:
        cv2.circle(frame, (startPos), startCircle[2],(0, 0, 255), 2)
        cv2.circle(frame, (startCircle[0],startCircle[1]), 5, (0, 0, 255), -1)  

    # Mark Exit Circle
    if left:
        cv2.circle(frame, (endPos), startCircle[2],(0, 0, 255), 2)
        cv2.circle(frame, (startCircle[0],startCircle[1]), 5, (0, 0, 255), -1)  

    if flipView:	
       frame = cv2.flip(frame, -1)
                                    
    # cv2.rectangle(frame, (0, 0), (640, 80), (0, 0, 0), -1)
    frame = drawAlphaRect(frame)

    cv2.putText(frame,"Start Ball",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,textcolor)
    cv2.putText(frame,"x:"+str(startCircle[0]),(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,textcolor)
    cv2.putText(frame,"y:"+str(startCircle[1]),(20,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,textcolor)

    if not lastShotSpeed == 0:
        cv2.putText(frame,"Last Shot",(400,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,textcolor,1)
        cv2.putText(frame,"Ball Speed: %.2f" % lastShotSpeed+" MPH",(400,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,textcolor,1)
        cv2.putText(frame,"HLA:  %.2f" % lastShotHLA+" Degrees",(400,80),cv2.FONT_HERSHEY_SIMPLEX,0.5,textcolor,1)
    
    if ballradius == 0:
        cv2.putText(frame,"radius:"+str(startCircle[2]),(20,80),cv2.FONT_HERSHEY_SIMPLEX,0.5,textcolor)
    else:
        cv2.putText(frame,"radius:"+str(startCircle[2])+" fixed at "+str(ballradius),(20,80),cv2.FONT_HERSHEY_SIMPLEX,0.5,textcolor)    

    cv2.putText(frame,"Actual FPS: %.2f" % fps,(200,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,textcolor)
    if overwriteFPS != 0:
        cv2.putText(frame,"Fixed FPS: %.2f" % overwriteFPS,(400,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,textcolor)
    else:
        cv2.putText(frame,"Detected FPS: %.2f" % video_fps[0],(400,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,textcolor)

    # ...

    outputframe = resizeWithAspectRatio(frame, width=int(args["resize"]))    
    cv2.imshow("Putting View: Press q to exit / a for adv. settings", outputframe)

    if args.get("debug", False):    
        # flip image on y-axis for view only
        if flipView:	
            mask = cv2.flip(mask, flipView)	
            origframe = cv2.flip(origframe, flipView)
        cv2.imshow("MaskFrame", mask)
        cv2.imshow("Original", origframe)


    # listen for keyboard events
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
    if key == ord("a"):

        if not a_key_pressed:
            cv2.namedWindow("Advanced Settings")
            if mjpegenabled != 0:
                vs.set(cv2.CAP_PROP_SETTINGS, 37)  
            cv2.resizeWindow("Advanced Settings", 1000, 440)
            cv2.createTrackbar("X Start", "Advanced Settings", int(sx1), 640, setXStart)
            cv2.createTrackbar("X End", "Advanced Settings", int(sx2), 640, setXEnd)
            cv2.createTrackbar("Y Start", "Advanced Settings", int(y1), 460, setYStart)
            cv2.createTrackbar("Y End", "Advanced Settings", int(y2), 460, setYEnd)
            cv2.createTrackbar("Radius", "Advanced Settings", int(ballradius), 50, setBallRadius)
            cv2.createTrackbar("Flip Image", "Advanced Settings", int(flipImage), 1, setFlip)
            cv2.createTrackbar("Flip View", "Advanced Settings", int(flipView), 1, setFlipView)
            cv2.createTrackbar("MJPEG", "Advanced Settings", int(mjpegenabled), 1, setMjpeg)
            cv2.createTrackbar("FPS", "Advanced Settings", int(overwriteFPS), 240, setOverwriteFPS)
            cv2.createTrackbar("Darkness", "Advanced Settings", int(darkness), 255, setDarkness)
            # cv2.createTrackbar("Saturation", "Advanced Settings", int(saturation), 255, setSaturation)
            # cv2.createTrackbar("Exposure", "Advanced Settings", int(exposure), 255, setExposure)
            a_key_pressed = True
        else:
            cv2.destroyWindow("Advanced Settings")

            exposure = vs.get(cv2.CAP_PROP_EXPOSURE)
            saturation = vs.get(cv2.CAP_PROP_SATURATION)
            autowb = vs.get(cv2.CAP_PROP_AUTO_WB)
            whiteBalanceBlue = vs.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U)
            whiteBalanceRed = vs.get(cv2.CAP_PROP_WHITE_BALANCE_RED_V)
            brightness = vs.get(cv2.CAP_PROP_BRIGHTNESS)
            contrast = vs.get(cv2.CAP_PROP_CONTRAST)
            hue = vs.get(cv2.CAP_PROP_HUE)
            gain = vs.get(cv2.CAP_PROP_GAIN)
            monochrome = vs.get(cv2.CAP_PROP_MONOCHROME)
            sharpness = vs.get(cv2.CAP_PROP_SHARPNESS)
            autoexposure = vs.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            gamma = vs.get(cv2.CAP_PROP_GAMMA)
            zoom = vs.get(cv2.CAP_PROP_ZOOM)
            focus = vs.get(cv2.CAP_PROP_FOCUS)
            autofocus = vs.get(cv2.CAP_PROP_AUTOFOCUS)


            debugLog("Saving Camera Settings to config.ini for restart")

            parser.set('putting', 'exposure', str(exposure))
            parser.set('putting', 'saturation', str(saturation))
            parser.set('putting', 'autowb', str(autowb))
            parser.set('putting', 'whiteBalanceBlue', str(whiteBalanceBlue))
            parser.set('putting', 'whiteBalanceRed', str(whiteBalanceRed))
            parser.set('putting', 'brightness', str(brightness))
            parser.set('putting', 'contrast', str(contrast))
            parser.set('putting', 'hue', str(hue))
            parser.set('putting', 'gain', str(gain))
            parser.set('putting', 'monochrome', str(monochrome))
            parser.set('putting', 'sharpness', str(sharpness))
            parser.set('putting', 'autoexposure', str(autoexposure))
            parser.set('putting', 'gamma', str(gamma))
            parser.set('putting', 'zoom', str(zoom))
            parser.set('putting', 'focus', str(focus))
            parser.set('putting', 'autofocus', str(autofocus))

            parser.write(open(CFG_FILE, "w"))

            a_key_pressed = False

    if key == ord("d"):
        if not d_key_pressed:
            args["debug"] = 1
            myColorFinder = ColorFinder(True)
            myColorFinder.setTrackbarValues(hsvVals)
            d_key_pressed = True
        else:
            args["debug"] = 0            
            myColorFinder = ColorFinder(False)
            cv2.destroyWindow("Original")
            cv2.destroyWindow("MaskFrame")
            cv2.destroyWindow("TrackBars")
            d_key_pressed = False


# close all windows
vs.release()
cv2.destroyAllWindows()
