from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("cars.mp4")
mask = cv2.imread("mask.jpg")
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("/Yolo-Weights/yolov8n.pt")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [184, 353, 691, 353]

totalCount = []

while True:
    success, img = cap.read()
    imgR = cv2.bitwise_and(img, mask)
    results = model(imgR, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w = x2-x1
            h = y2-y1
            bbox = int(x1), int(y1), int(w), int(h)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            conf = math.ceil((box.conf[0]*100))/100
            currentClass = model.names[int(box.cls)]
            if currentClass == "car" or currentClass == "true" or currentClass == "bus" or currentClass == "motorbike" and conf > 3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0,x1), max(45,y1)), scale=1, thickness=1)
                cvzone.cornerRect(img, bbox, l=9, rt=5, colorR=(255, 0, 0))
                curr = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, curr))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,0,255), 5)
    for results in resultsTracker:
        x1, y1, x2, y2, ID = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(results)
        w,h = int(x2-x1), int(y2-y1)
        # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{int(ID)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), 5, (255,0,255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1]-30 < cy < limits[3]+30:

            if totalCount.count(ID) == 0:
                totalCount.append(ID)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50,50))
    cv2.imshow("Image", img)
    cv2.waitKey(1)
