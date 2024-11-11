import cv2
import numpy as np

# Video
# video = cv2.VideoCapture("Pedestrians.mp4")
# video = cv2.VideoCapture("Tokyo_Neighborhood_Walking1.mp4")
# video = cv2.VideoCapture("Tesla_Driving1.mp4")
video = cv2.VideoCapture("Tokyo_Evening_Walk1.mp4")
# video = cv2.VideoCapture("Tokyo_Evening_Walk2.mp4")
# video = cv2.VideoCapture("Highway.mp4")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

while True:
    (read_successful, frame) = video.read()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.25:
            idx = int(detections[0, 0, i, 1])
            if (idx == 2 or idx == 7 or idx == 14 or idx == 15):
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                print(idx)
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                print("[INFO] {}".format(label))
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    cv2.imshow("Tye Maxson Cars and Pedestrians Detector",frame)
    key = cv2.waitKey(1)
    if key==81 or key==113:
        break
