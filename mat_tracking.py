from collections import deque
import cv2
import numpy as np
import time
import argparse
from configparser import ConfigParser
from ColorModuleExtended import ColorFinder
import ast
import imutils


from aruco_detect import *
from ball_colors import *
from ball_utils import *
from image_utils import *

bufferSize=64
textcolor=(200, 200, 200)
videoWidth=640
ballradius = 0
darkness = 0
d_key_pressed = False 
startminimum = 30
golfballradius = 21.33; # in mm

started = False
entered = False
left = False
startCandidates = []
noOfStarts = 0
lastShotStart = (0,0)
lastShotEnd = (0,0)
lastShotSpeed = 0
lastShotHLA = 0 
ballradius = 0
startPos = (0,0)
endPos = (0,0)
x = 0
y = 0
radius = 0
center= (0,0)
startCircle = (0, 0, 0)
endCircle = (0, 0, 0)
pts = deque()
tims = deque()
tim1 = 0
tim2 = 0

ap = argparse.ArgumentParser()
ap.add_argument("--http",
                help="Send http requests to this URL (e.g. --http http://localhost:8888/putting)")
ap.add_argument("--config",
                help="Path to config.ini file")
ap.add_argument("-d", "--debug",
                help="debug - color finder and wait timer")

args = vars(ap.parse_args())

# parse http request url
httpRequestUrl = False
if args.get("http", False):
    httpRequestUrl = args['http']

# parse http request url
configFilePath = 'config.ini'
if args.get("config", False):
    configFilePath = args['config']
parser = ConfigParser()
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
if parser.has_option('putting', 'customhsv'):
    customhsv=ast.literal_eval(parser.get('putting', 'customhsv'))
    debugLog(customhsv)
    hsvVals = customhsv
else:
    customhsv={}
    hsvVals=orange


# Detection Gateway
# x1=sx2+10
# x2=x1+10

#coord of polygon in frame::: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
# startcoord=[[sx1,y1],[sx2,y1],[sx1,y2],[sx2,y2]]
startcoord=None
#coord of polygon in frame::: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
# coord=[[x1,y1],[x2,y1],[x1,y2],[x2,y2]]

def GetAngle (p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dX = x2 - x1
    dY = y2 - y1
    rads = math.atan2 (-dY, dX)

    # if flipImage == 1 and videofile == False:    	
    #     rads = rads*-1
    return math.degrees (rads)

def columnMax(points, column):
  return [max(i) for i in zip(*points)][column]

def columnMin(points, column):
  return [min(i) for i in zip(*points)][column]

def boundingBox(points):
  # The upperleft point is the smallest of all X values and the smallest of all Y values
  # upperleftXY = [points.column(0).min, points.column(1).max]
  upperLeft = [columnMin(points, 0), columnMin(points, 1)]

  # The lowerrightXY point is the largest of all X values and largest of all Y values
  # lowerRight = [points.column(0).max, points.column(1).min]
  lowerRight = [columnMax(points, 0), columnMax(points, 1)]

  upperRight = [lowerRight[0], upperLeft[1]]
  lowerLeft = [upperLeft[0], lowerRight[1]]

  return [upperLeft, upperRight, lowerLeft, lowerRight]


splashImagePath = resource_path("images/splash.png")
debugLog(splashImagePath)
splashFrame = cv2.imread(splashImagePath)
# origframe2 = cv2.imread(splashImagePath)
cv2.putText(splashFrame, "Starting Video: Try MJPEG option in advanced settings for faster startup", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textcolor)
outputframe = resizeWithAspectRatio(splashFrame, videoWidth)
cv2.imshow("Putting View: Press q to exit / a for adv. settings", outputframe)

webcamindex = 0
mjpegenabled = 0
ps4=0

if args.get("camera", False):
    webcamindex = args["camera"]
    debugLog("Putting Cam activated at "+str(webcamindex))

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

video_fps = vs.get(cv2.CAP_PROP_FPS)
height = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = vs.get(cv2.CAP_PROP_FRAME_WIDTH)


time.sleep(0.5)

arucoDictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
arucoParameters =  cv2.aruco.DetectorParameters()
arucoDetector = cv2.aruco.ArucoDetector(arucoDictionary, arucoParameters)
lastDetected = None

if args.get("debug", False):
    myColorFinder = ColorFinder(True)
    myColorFinder.setTrackbarValues(hsvVals)
else:
    myColorFinder = ColorFinder(False)

mainWindowTitle = "Putting View: Press q to exit / a for adv. settings"

while True:
  
  frameTime = time.time()
  # read the frame
  ret, frame = vs.read()

  if frame is None:
    debugLog("no frame!")
    frame = cv2.imread(splashImagePath)
    cv2.putText(frame,"Error: "+"No Frame",(20, 20),cv2.FONT_HERSHEY_SIMPLEX,0.5,textcolor)
    cv2.putText(frame,"Message: "+message,(20, 40),cv2.FONT_HERSHEY_SIMPLEX,0.5,textcolor)
    cv2.imshow(mainWindowTitle, frame)
    cv2.waitKey(0)
    break

  # origframe = frame.copy()
  cv2.normalize(frame, frame, 0-darkness, 255-darkness, norm_type=cv2.NORM_MINMAX)
  frame = imutils.resize(frame, width=640, height=360)

  # # locate ArUco markers on putting mat
  # (corners, ids, rejected) = arucoDetector.detectMarkers(frame)

  # detectedPoints = []
  # # verify *at least* one ArUco marker was detected
  # if len(corners) > 0:
  #   # flatten the ArUco IDs list
  #   ids = ids.flatten()
  #   # loop over the detected ArUCo corners
  #   for (markerCorner, markerID) in zip(corners, ids):
  #     # extract the marker corners (which are always returned in
  #     # top-left, top-right, bottom-right, and bottom-left order)
  #     corners = markerCorner.reshape((4, 2))
  #     (topLeft, topRight, bottomRight, bottomLeft) = corners
  #     # convert each of the (x, y)-coordinate pairs to integers
  #     topRight = (int(topRight[0]), int(topRight[1]))
  #     bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
  #     # compute and draw the center (x, y)-coordinates of the ArUco
  #     # marker
  #     cX = int((topLeft[0] + bottomRight[0]) / 2.0)
  #     cY = int((topLeft[1] + bottomRight[1]) / 2.0)
  #     cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
  #     detectedPoints.append([cX, cY])


  detectedPoints = detectArucoPoints(arucoDetector, frame)
  if len(detectedPoints) == 2:
    lastDetected = frameTime
    result = boundingBox(detectedPoints)
    if len(result) == 4:
      startcoord = result
      x2 = startcoord[1][0]+20
      coord=[
        [startcoord[1][0],startcoord[1][1]],
        [x2,startcoord[1][1]],
        [startcoord[1][0],startcoord[3][1]],
        [x2,startcoord[3][1]]]

  if lastDetected != None:
    timeSinceLastDetected = frameTime - lastDetected
    if timeSinceLastDetected > 2000:
      startcoord = None

  contourMask = None
  if startcoord != None:


    # once we have our start position set we can start looking for a ball
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Find the Color Ball
    imgColor, mask, newHSV = myColorFinder.update(hsv, hsvVals)
    # debugLog(hsvVals)
    # debugLog(newHSV)
    if hsvVals != newHSV:
        debugLog(newHSV)
        parser.set('putting', 'customhsv', str(newHSV))
        parser.write(open(configFilePath, "w"))
        hsvVals = newHSV
        debugLog("HSV values changed - Set custom color to config.ini")

    # sx1=startcoord[0][0]
    # sx2=startcoord[2][0]
    # sy1=startcoord[0][1]
    # sy2=startcoord[2][1]
    sx1 = startcoord[0][0]
    sx2 = startcoord[1][0]
    sy1 = startcoord[0][1]
    sy2 = startcoord[3][1]

    # print(startcoord)
    # mask = mask[124:298, 192:640]
    # mask = mask[124:298, 192:640]
    pad = 0
    mask = mask[startcoord[0][1]+pad:startcoord[2][1], startcoord[0][0]:640]

    # contourMask = mask
    # cv2.imshow("MaskFrame", compframe)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    center = None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # # only proceed if at least one contour was found
    # print(startcoord)
    if len(contours) > 0:
      # result = detectBall(
      #   frame,
      #   contours,
      #   mask,
      #   startcoord,
      #   started,
      #   entered,
      #   startPos,
      #   endPos,
      #   left,
      #   startCandidates,
      #   noOfStarts,
      #   lastShotStart,
      #   lastShotEnd,
      #   lastShotSpeed,
      #   lastShotHLA,
      #   frameTime,
      #   ballradius
      # )
      # if result != None:
      #   (started, entered, startPos, endPos, startCircle, endCircle) = result
      #   # print(started, startCircle, endCircle)
      for index in range(len(contours)):
        circle = (0,0,0)
        center= (0,0)
        radius = 0
        # Eliminate countours that are outside the y dimensions of the detection zone
        ((tempcenterx, tempcentery), tempradius) = cv2.minEnclosingCircle(contours[index])
        tempcenterx = tempcenterx + sx1
        tempcentery = tempcentery + sy1

        if (tempcentery >= sy1 and tempcentery <= sy2):
            rangefactor = 50
            cv2.drawContours(mask, contours, index, (60, 255, 255), 1)
            # cv2.putText(frame,"Radius:"+str(int(tempradius)),(int(tempcenterx)+3, int(tempcentery)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))
            # Eliminate countours significantly different than startCircle by comparing radius in range
            if (started == True and startCircle[2]+rangefactor > tempradius and startCircle[2]-rangefactor < tempradius):
                x = int(tempcenterx)
                y = int(tempcentery)
                radius = int(tempradius)
                center = (x,y)
                # return(center, radius)
            else:
                if not started:
                    x = int(tempcenterx)
                    y = int(tempcentery)
                    radius = int(tempradius)
                    center = (x,y)
                    # return(center, radius)
                    #print("No Startpoint Set Yet: "+str(center)+" "+str(startCircle[2]+rangefactor)+" > "+str(radius)+" AND "+str(startCircle[2]-rangefactor)+" < "+str(radius))
        else:
            break

        if radius >=5:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points  
            circle = (x,y,radius)
            wiggleRoom = 10
            if circle:
                # check if the circle is stable to detect if a new start is there
                if not started or startPos[0]+wiggleRoom <= center[0] or startPos[0]-wiggleRoom >= center[0]:
                    # print("startPos[0]+wiggleRoom, center[0]: "+str(startPos[0]+wiggleRoom)+","+str(center[0]))
                    # print("startPos[0]-wiggleRoom, center[0]: "+str(startPos[0]-wiggleRoom)+","+str(center[0]))
                    if (center[0] >= sx1 and center[0] <= sx2):
                        startCandidates.append(center)
                        # print("Found a candidate "+str(len(startCandidates)))
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
                                print("New ball start position found!")
                                eventData = { "eventName": "ready", "data": { "position": center } }
                                sendEvent(eventData)
                                if httpRequestUrl != False:
                                    makeHTTPRequest(httpRequestUrl, eventData)
                                # replayavail = False
                                noOfStarts = noOfStarts + 1
                                lastShotSpeed = 0
                                pts.clear()
                                tims.clear()
                                filteredcircles = []
                                filteredcircles.append(circle)
                                startCircle = circle
                                startPos = center
                                print("SET START POS"+str(center))
                                startTime = frameTime
                                #print("Start Position: "+ str(startPos[0]) +":" + str(startPos[1]))
                                # Calculate the pixel per mm ratio according to z value of circle and standard radius of 2133 mm

                                if ballradius == 0:
                                    pixelmmratio = circle[2] / golfballradius
                                else:
                                    pixelmmratio = ballradius / golfballradius

                                started = True
                                entered = False
                                left = False
                                # update the points and tims queues
                                pts.appendleft(center)
                                tims.appendleft(frameTime)  

                                # replay1 = cv2.VideoWriter('replay1/Replay1_'+ str(noOfStarts) +'.mp4', apiPreference=0, fourcc=fourcc,fps=120, frameSize=(int(width), int(height)))
                                # if replaycam == 1:
                                #     replay2 = cv2.VideoWriter('replay2/Replay2_'+ str(noOfStarts) +'.mp4', apiPreference=0, fourcc=fourcc,fps=120, frameSize=(int(replaycamwidth), int(replaycamheight)))

                    else:

                        if (x >= coord[0][0] and entered == False and started == True):
                            cv2.line(frame, (coord[0][0], coord[0][1]), (coord[2][0], coord[2][1]), (0, 255, 0),2)  # Changes line color to green
                            tim1 = frameTime
                            print("Ball Entered. Position: "+str(center))
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
                                    print("Ball Left. Position: "+str(center))
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
                                        print("Time Elapsed in Sec: "+str(timeElapsedSeconds))
                                        print("Distance travelled in MM: "+str(distanceTraveledMM))
                                        print("Speed: "+str(speed)+" MPH")
                                        # update the points and tims queues
                                        pts.appendleft(center)
                                        tims.appendleft(frameTime)
                                        break
                                else:
                                    print("False Exit after the Ball")

                                    # flip image on y-axis for view onl
        
    # loop over the set of tracked points
    if len(pts) != 0 and entered == True:
      for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(bufferSize / float(i + 1)) * 1.5)
      timeSinceEntered = (frameTime - tim1)
      # replaytrigger = tim1
    
    if left == True:

        # Send Shot Data
        if (tim2 and timeSinceEntered > 0.5 and distanceTraveledMM and timeElapsedSeconds and speed >= 0.5 and speed <= 25):
            print("----- Shot Complete --------")
            print("Time Elapsed in Sec: "+str(timeElapsedSeconds))
            print("Distance travelled in MM: "+str(distanceTraveledMM))
            print("Speed: "+str(speed)+" MPH")

            #     ballSpeed: ballData.BallSpeed,
            #     totalSpin: ballData.TotalSpin,
            totalSpin = 0
            #     hla: ballData.LaunchDirection,
            launchDirection = (GetAngle((startCircle[0],startCircle[1]),endPos)*-1)
            print("HLA: Line"+str((startCircle[0],startCircle[1]))+" Angle "+str(launchDirection))
            #Decimal(launchDirection);
            if (launchDirection > -40 and launchDirection < 40):

                lastShotStart = (startCircle[0],startCircle[1])
                lastShotEnd = endPos
                lastShotSpeed = speed
                lastShotHLA = launchDirection
                    
                # Data that we will send in post request.
                data = {"ballSpeed":"%.2f" % speed,"totalSpin":totalSpin,"launchDirection":"%.2f" % launchDirection}
                eventData = { "eventName": "putt", "data": data }
                # # The POST request to our node server
                # if args["ballcolor"] == "calibrate":
                #     debugLog("calibration mode - shot data not send")
                # else:
                sendEvent(eventData)
                if httpRequestUrl != False:
                    makeHTTPRequest(httpRequestUrl, eventData)
            else:
                print("Misread on HLA - Shot not send!!!")    
            # if len(pts) > calObjectCount:
            #     calObjectCount = len(pts)
            print("----- Data reset --------")
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
            print("----- Data reset --------")
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

    frame = drawMaskOnFrame(frame, mask)
    # draw startpoint zone
    cv2.line(frame, (startcoord[0][0], startcoord[0][1]), (startcoord[1][0], startcoord[1][1]), (0, 210, 255), 2)  # First horizontal line
    cv2.line(frame, (startcoord[0][0], startcoord[0][1]), (startcoord[2][0], startcoord[2][1]), (0, 210, 255), 2)  # Vertical left line
    cv2.line(frame, (startcoord[2][0], startcoord[2][1]), (startcoord[3][0], startcoord[3][1]), (0, 210, 255), 2)  # Second horizontal line
    cv2.line(frame, (startcoord[1][0], startcoord[1][1]), (startcoord[3][0], startcoord[3][1]), (0, 210, 255), 2)  # Vertical right line

    # Detection Gateway
    cv2.line(frame, (coord[0][0], coord[0][1]), (coord[1][0], coord[1][1]), (0, 0, 255), 2)  # First horizontal line
    cv2.line(frame, (coord[0][0], coord[0][1]), (coord[2][0], coord[2][1]), (0, 0, 255), 2)  # Vertical left line
    cv2.line(frame, (coord[2][0], coord[2][1]), (coord[3][0], coord[3][1]), (0, 0, 255), 2)  # Second horizontal line
    cv2.line(frame, (coord[1][0], coord[1][1]), (coord[3][0], coord[3][1]), (0, 0, 255), 2)  # Vertical right line    
  else:
    print("No start coordinates found!")


  # draw GUI / debug text
  frame = drawAlphaRect(frame, 640, 40)
  # cv2.putText(frame, "Start Ball", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textcolor)

  outputframe = resizeWithAspectRatio(frame, videoWidth)    
  cv2.imshow(mainWindowTitle, outputframe)


  # if args.get("debug", False):    
  #   if contourMask.size > 0:
  #     cv2.imshow("MaskFrame", contourMask)
  #   else:
  #     cv2.imshow("MaskFrame", splashFrame)

  key = cv2.waitKey(1) & 0xFF
  # if the 'q' key is pressed, stop the loop
  if key == ord("q"):
      break
  if key == ord("d"):
      if not d_key_pressed:
          args["debug"] = 1
          myColorFinder = ColorFinder(True)
          myColorFinder.setTrackbarValues(hsvVals)
          d_key_pressed = True
      else:
          args["debug"] = 0            
          myColorFinder = ColorFinder(False)
          cv2.destroyWindow(myColorFinder.windowName)
          d_key_pressed = False


# close all windows
vs.release()
cv2.destroyAllWindows()
