# package imports
import numpy as np
import time
import imutils
import cv2
import os
import argparse


base_confidence = 0.5 #weak detections
base_threshold = 0.3 #segmentation threshold for masks
IGNORE = set(["person"]) #list of labels that won't be ignored later


# import the video file location from command line
argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "--input", required=True) # file must be specified
argParser.add_argument("-o", "--output", required=True) # output video file
args = vars(argParser.parse_args()) # parse arguments

#preset input paths for weights, labels and configuration.
LABELS = open("./mask-rcnn-coco/object_detection_classes_coco.txt").read().strip().split("\n")
weightsPath = "./mask-rcnn-coco/frozen_inference_graph.pb"
configPath =  "./mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"


np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")


#load trained weights
print("Loading trained weights")
weightNet = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)


vs = cv2.VideoCapture(args["input"])
writer = None


#get number of frames in video
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))
except:
	print("[INFO] could not determine # of frames in video")
	total = -1

# loop over each frame in the video
while True:
	# read the next frame, if fails => reached end of video
	(grabbed, frame) = vs.read()
	if not grabbed:
		break

	#first forward-pass of mask R-CNN
	blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
	weightNet.setInput(blob)
	start = time.time()
	(boxes, masks) = weightNet.forward(["detection_out_final","detection_masks"])
	end = time.time()

	# boxes capture objects, loop over objects
	for i in range(0, boxes.shape[2]):
		classID = int(boxes[0, 0, i, 1]) #type of object
		confidence = boxes[0, 0, i, 2] #confidence

		if confidence > base_confidence: #filter unlikely options
			
			if LABELS[classID] not in IGNORE: #skip labels that are not 'person'
				continue

			#rescaling of bounding box
			(H, W) = frame.shape[:2]
			box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = box.astype("int")
			boxW = endX - startX
			boxH = endY - startY

			#segment image pixel-wise
			mask = masks[i, classID]
			mask = cv2.resize(mask, (boxW, boxH),
				interpolation=cv2.INTER_CUBIC) #INTER_CUBIC beneficial over INTER_NEAREST
			mask = (mask > base_threshold)

			# extract the masked ROI of the image
			roi = frame[startY:endY, startX:endX][mask]

			#paint the mask in a particular colour
			color = COLORS[classID]
			blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

			# store the blended ROI in the original frame
			frame[startY:endY, startX:endX][mask] = blended

			# draw the bounding box of the instance on the frame
			color = [int(c) for c in color]
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				color, 2)

			# draw the predicted label and associated probability of
			# the instance segmentation on the frame
			text = "{}: {:.4f}".format(LABELS[classID], confidence)
			cv2.putText(frame, text, (startX, startY - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()