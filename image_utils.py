import cv2
import math
import imutils

def drawMaskOnFrame(frame, mask):
  compframe = frame.copy()
  maskMerged = cv2.merge((mask,mask,mask))

  frameH = frame.shape[0]-1
  compW = maskMerged.shape[1]-1
  compH = maskMerged.shape[0]-1
  compNewW = 200
  compNewH = math.floor(compH * (compNewW/compW))
  # draw in bottom left corner
  compY = frameH - compNewH
  maskMerged = imutils.resize(maskMerged, width=compNewW, height=compNewH)
  compframe[compY:compY+compNewH,0:compNewW,:] = maskMerged[0:compNewH,0:compNewW,:]
  return compframe


def drawAlphaRect(image, width=640, height=90):
    overlay = image.copy()
    x, y, w, h = 0, 0, width, height  # Rectangle parameters
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 0), -1)  # A filled rectangle
    alpha = 0.4  # Transparency factor.
    # Following line overlays transparent rectangle over the image
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

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
