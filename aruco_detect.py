import cv2

def detectArucoPoints(arucoDetector, frame):
  # locate ArUco markers on putting mat
  (corners, ids, rejected) = arucoDetector.detectMarkers(frame)

  detectedPoints = []
  # verify *at least* one ArUco marker was detected
  if len(corners) > 0:
    # flatten the ArUco IDs list
    ids = ids.flatten()
    # loop over the detected ArUCo corners
    for (markerCorner, markerID) in zip(corners, ids):
      # extract the marker corners (which are always returned in
      # top-left, top-right, bottom-right, and bottom-left order)
      corners = markerCorner.reshape((4, 2))
      (topLeft, topRight, bottomRight, bottomLeft) = corners
      # convert each of the (x, y)-coordinate pairs to integers
      topRight = (int(topRight[0]), int(topRight[1]))
      bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
      # compute and draw the center (x, y)-coordinates of the ArUco
      # marker
      cX = int((topLeft[0] + bottomRight[0]) / 2.0)
      cY = int((topLeft[1] + bottomRight[1]) / 2.0)
      cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
      detectedPoints.append([cX, cY])
  return detectedPoints