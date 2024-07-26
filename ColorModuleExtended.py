"""
Color Module - original by cvzone - changed to set values
Finds color in an image based on hsv values
Can run as stand alone to find relevant hsv values

"""

import cv2
import numpy as np
import logging



class ColorFinder:
    def __init__(self, trackBar=False):
        self.trackBar = trackBar
        self.windowName = "Mask Editor"
        if self.trackBar:
            self.initTrackbars()
            trackBar.__setattr__

    def empty(self, a):
        pass

    def destroyWindow():
        cv2.destroyWindow(self.windowName)

    def initTrackbars(self):
        """
        To intialize Trackbars . Need to run only once
        """
        
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.windowName, 640, 240)
        cv2.moveWindow(self.windowName, 20, 20)
        cv2.createTrackbar("Hue Min", self.windowName, 0, 179, self.empty)
        cv2.createTrackbar("Hue Max", self.windowName, 179, 179, self.empty)
        cv2.createTrackbar("Sat Min", self.windowName, 0, 255, self.empty)
        cv2.createTrackbar("Sat Max", self.windowName, 255, 255, self.empty)
        cv2.createTrackbar("Val Min", self.windowName, 0, 255, self.empty)
        cv2.createTrackbar("Val Max", self.windowName, 255, 255, self.empty)

    def getTrackbarValues(self):
        """
        Gets the trackbar values in runtime
        :return: hsv values from the trackbar window
        """

        hmin = cv2.getTrackbarPos("Hue Min", self.windowName)
        smin = cv2.getTrackbarPos("Sat Min", self.windowName)
        vmin = cv2.getTrackbarPos("Val Min", self.windowName)
        hmax = cv2.getTrackbarPos("Hue Max", self.windowName)
        smax = cv2.getTrackbarPos("Sat Max", self.windowName)
        vmax = cv2.getTrackbarPos("Val Max", self.windowName)

        hsvVals = {"hmin": hmin, "smin": smin, "vmin": vmin,
                   "hmax": hmax, "smax": smax, "vmax": vmax}
        
        return hsvVals

    def setTrackbarValues(self, hsvValues):
        """
        Sets the trackbar values in runtime
        """

        hmin = cv2.setTrackbarPos("Hue Min", self.windowName,hsvValues['hmin'])
        smin = cv2.setTrackbarPos("Sat Min", self.windowName,hsvValues['smin'])
        vmin = cv2.setTrackbarPos("Val Min", self.windowName,hsvValues['vmin'])
        hmax = cv2.setTrackbarPos("Hue Max", self.windowName,hsvValues['hmax'])
        smax = cv2.setTrackbarPos("Sat Max", self.windowName,hsvValues['smax'])
        vmax = cv2.setTrackbarPos("Val Max", self.windowName,hsvValues['vmax'])

        hsvVals = {"hmin": hmin, "smin": smin, "vmin": vmin,
                   "hmax": hmax, "smax": smax, "vmax": vmax}
        
        return hsvVals

    def update(self, img, myColor=None):
        """
        :param img: Image in which color needs to be found
        :param hsvVals: List of lower and upper hsv range
        :return: (mask) bw image with white regions where color is detected
                 (imgColor) colored image only showing regions detected
        """
        imgColor = [],
        mask = []

        if self.trackBar:
            myColor = self.getTrackbarValues()

        # if isinstance(myColor, str):
        #     myColor = self.getColorHSV(myColor)

        if myColor is not None:
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([myColor['hmin'], myColor['smin'], myColor['vmin']])
            upper = np.array([myColor['hmax'], myColor['smax'], myColor['vmax']])
            mask = cv2.inRange(imgHSV, lower, upper)
            imgColor = cv2.bitwise_and(img, img, mask=mask)
        return imgColor, mask, myColor

    def getColorHSV(self, myColor):

        if myColor == 'red':
            output = {'hmin': 146, 'smin': 141, 'vmin': 77, 'hmax': 179, 'smax': 255, 'vmax': 255}
        elif myColor == 'green':
            output = {'hmin': 44, 'smin': 79, 'vmin': 111, 'hmax': 79, 'smax': 255, 'vmax': 255}
        elif myColor == 'blue':
            output = {'hmin': 103, 'smin': 68, 'vmin': 130, 'hmax': 128, 'smax': 255, 'vmax': 255}
        else:
            output = None
            logging.warning("Color Not Defined")
            logging.warning("Available colors: red, green, blue ")

        return output


def main():
    myColorFinder = ColorFinder(False)
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Custom Orange Color
    hsvVals = {'hmin': 10, 'smin': 55, 'vmin': 215, 'hmax': 42, 'smax': 255, 'vmax': 255}

    while True:
        success, img = cap.read()
        imgRed, _ = myColorFinder.update(img, "red")
        imgGreen, _ = myColorFinder.update(img, "green")
        imgBlue, _ = myColorFinder.update(img, "blue")
        imgOrange, _ = myColorFinder.update(img, hsvVals)

        cv2.imshow("Red", imgRed)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
