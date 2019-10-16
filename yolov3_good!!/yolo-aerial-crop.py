# USAGE
# python yolo-aerial-crop.py -y yolo-coco

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import imagezmq
from PIL import Image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "aerial.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3-aerial.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3-tiny-aerial.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

image_hub = imagezmq.ImageHub()
print('[Server] is Activate!')
while True:
    rpi_name, img = image_hub.recv_image()
    (tempH, tempW) = img.shape[:2]
    # if Mac_Camera filmed, the size is (720, 1280).
    if tempH == 720: #[top, bottom, right, left]
        #image = [img[0:256, 512:720], img[480:720, 512:720], img[256:480, 0:512], img[256:480, 720:1280]]
        image = [img[0:256, 512:616], img[480:720, 616:720], img[368:480, 0:512], img[256:368, 720:1280]]
    # if PI_Camera f filmed, the size is (900, 1600).
    elif tempH == 900: #[top, bottom, right, left]
        image = [img[0:320, 640:900], img[600:900, 640:900], img[320:600, 0:640], img[320:600, 900:1600]]
    else:
        image = [img[0:120, 0:160], img[120:240, 0:160], img[0:120, 160:320], img[120:240, 160:320]]
    """#image crop test #if you want delete it, whatever you want to do it.
    image = [img[0:tempH/2, 0:tempW/2], img[0:tempH/2, tempW/2:tempW], img[tempH/2:tempH, 0:tempW/2], img[tempH/2:tempH, tempW/2:tempW]]"""
    #
    carNum = [0, 0, 0, 0]
    for imgNum in range(0,4):
        # load our input image and grab its spatial dimensions
        (H, W) = image[imgNum].shape[:2]

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image[imgNum], 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        """start = time.time()"""
        layerOutputs = net.forward(ln)
        """end = time.time()"""

        # show timing information on YOLO
        """print("[INFO] YOLO took {:.6f} seconds".format(end - start))"""

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
            args["threshold"])

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image[imgNum], (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                if text[0] == 'c' and text[1] == 'a' and text[2] == 'r':
                    carNum[imgNum] = carNum[imgNum] + 1
                elif text[0] == 'b' and text[1] == 'u' and text[2] == 's':
                    carNum[imgNum] = carNum[imgNum] + 1
                elif text[0] == 'm' and text[1] == 'i' and text[2] == 'n' and text[3] == 'i' and text[4] == 'b' and text[5] == 'u' and text[6] == 's':
                    carNum[imgNum] = carNum[imgNum] + 1
                cv2.putText(image[imgNum], text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        """
        # show the output image
        cv2.namedWindow(rpi_name + str(imgNum), cv2.WINDOW_NORMAL) # You can resize the window
        cv2.imshow(rpi_name + str(imgNum), image[imgNum])
        print("carNum[" + str(imgNum) + "] : ", carNum[imgNum])
        """
        
    # show the output image
    cv2.namedWindow(rpi_name+" top", cv2.WINDOW_NORMAL) # You can resize the window
    cv2.namedWindow(rpi_name+" bottom", cv2.WINDOW_NORMAL) # You can resize the window
    cv2.namedWindow(rpi_name+" right", cv2.WINDOW_NORMAL) # You can resize the window
    cv2.namedWindow(rpi_name+" left", cv2.WINDOW_NORMAL) # You can resize the window
    
    cv2.imshow(rpi_name+" top", image[0])
    cv2.imshow(rpi_name+" bottom", image[1])
    cv2.imshow(rpi_name+" right", image[2])
    cv2.imshow(rpi_name+" left", image[3])
    
    print("-----carNum-----")
    print(" top    : " + str(carNum[0]))
    print(" bottom : " + str(carNum[1]))
    print(" right  : " + str(carNum[2]))
    print(" left   : " + str(carNum[3]))
    
    ###
    if cv2.waitKey(1) == ord('q'):
        break
    
    image_hub.send_reply(b'OK')
