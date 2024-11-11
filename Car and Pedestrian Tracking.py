import cv2
import numpy as np

# Video that is being used in the progrma
# video = cv2.VideoCapture("Pedestrians.mp4")
# video = cv2.VideoCapture("Tesla_Driving1.mp4")
video = cv2.VideoCapture("Tokyo_Evening_Walk1.mp4")
# video = cv2.VideoCapture("Highway.mp4")

# Different classes used from the improted neural network
# If classes is not here then a number is just displayed and nothng is displayed on the acutal classification
# Converts the numbers given by the program from the nurtal network into a class that can be understood
CLASSES = ["background", "airplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "dining_table",
	"dog", "horse", "motorcycle", "person", "potted_plant", "sheep",
	"sofa", "train", "tvmonitor"]

# The different colors that are used for the boxes
# Colors become random but are the same throughtout the entire run of the program
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Importing the neutal network form the two files
# (Premade)
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

# While loop to keep everything working throughtout the entire video/video stream
while True:
    # Current Frame Reading

    # Currently not being used since the original neual network is not being used
    # Reading the video from the downloaded video in same folder
    (read_successful, frame) = video.read()
    # What happens when the video is read successfully
    if read_successful:
        # Converting to grayscale
        grayscaled_frames = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break
    
    # Getting the original height and width of the video
    # From the frame.shape of the given video or photo
    (h, w) = frame.shape[:2]

    # Changing the entire video into a "blob" making the video a lower resolution
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Changes the set input into the blob
    net.setInput(blob)

    # This is where the detections are being made
    # From the net.forard() is where the detections are detected
    detections = net.forward()

    # loop over the detections
    # Taking the i that is found in the aove detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        # The detections in the above are then set equal to a confidence level in decimal form (0.5, 0.9, or 1(as a 100))
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        # In this case of the confidence of the detection is greater than 25% then it goes throught the if statement
        if confidence > 0.25:
            # Ectracts the index of the class from detections, then computes the (x, y) coordinated of the bounding box
            # The bounding box is then able to be put into the frame thus outlining the object that it is detecting
            idx = int(detections[0, 0, i, 1])
            # This is for the program to know that when the objects that are beign detected are on of these number
            # These numbers are from the class list above
            # Only if the idx from the detections are in the statement below will the program put a box around the object
            if (idx == 1 or idx == 2 or idx == 7 or idx == 14 or idx == 15):
                # This defines where the box goes with the different detections and the np.array
                # The no.array has the width and height of the original image
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                # This defines what the box will be using the starting X and Y and the ending X and Y axis
                (startX, startY, endX, endY) = box.astype("int")
                # display the prediction
                # The following would display the the idx of the detections from above
                print(idx) # Possible do not need if I do not want to print out the idx before printing the info
                # This is where the program labels what objects it is detecting
                # It get formatted in the Classes list to classify the objects correctly
                # The cofidence of the detection is multipled by 100 to get it in percentage instead of decimal
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                # The info gets printed in the individual frames labeling what it is classifing
                print("[INFO] {}".format(label))
                # The rectangeles are finally drawn on the different frames
                # It uses the fame of each video and the starting x and y with the ending x and y
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    # The colors are then set for the different rectangels 
                    COLORS[idx], 2)
                # This makes it so that when the rectangles is drawn on it does not just on the object 
                # it makes it so that it draws just a bit away from the object ot keep it in frame
                y = startY - 15 if startY - 15 > 15 else startY + 15
                # This is what put the rectangle over the fram along with the label starting fromt eh starting X and Y
                cv2.putText(frame, label, (startX, y),
                    # This is where the font of the labels are defined
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # Showing video with overlays
    # This is where everything comes together
    # The show from the terminal or command prompt
    cv2.imshow("Tye Maxson - Cars and Pedestrians Detector",frame)
    
    # Stopping AutoClose
    # This allows you to close the terminal when the video is still playing without having to go throught the entire thing
    key = cv2.waitKey(1)

    # Stopping Program
    # Defines the keyes that can beused to terminate the program
    # Using the key q or capitalized Q
    if key==81 or key==113:
        # Breaks the program thus exiting the program
        break
