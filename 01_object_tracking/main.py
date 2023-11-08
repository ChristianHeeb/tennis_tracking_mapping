# Code Ideas from tutorial: https://www.youtube.com/watch?v=O3b8lVF93jU&list=PL9kP7DArSEpeXavXusoRckIYr102raPu8&index=1&ab_channel=Pysource

import cv2
import csv
from tracker import *


# Create tracker object
tracker = EuclideanDistTracker()

# Initialize CSV file
with open("object_positions.csv", "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header
    csvwriter.writerow(["Frame", "ID", "X", "Y", "W", "H"])

cap = cv2.VideoCapture("/Users/christianheeb/Desktop/tennis_tracking_mapping/99_pictures_videos/tennis_match2.mp4")

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=100)

frame_count = 0  # To keep track of the frame number

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape

    # Extract Region of interest 
    # e.g. only use bottom part of Video
    roi = frame[400:1500, 1:1500]

    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 800:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(
            roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2
        )
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Save positions to CSV
        with open("object_positions.csv", "a", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([frame_count, id, x, y, w, h])

    frame_count += 1  # Increment frame number

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)


    key = cv2.waitKey(30)
    if key == 27: # press ESC key to exit     
        break

cap.release()
cv2.destroyAllWindows()
