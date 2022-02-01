# Import the necessary packages
import os
import shutil
import cv2
import numpy as np
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array
import functools
from  de_skew import deskew
 
###############################################
# This takes two stagesclea
# The first stage is to segment characters
# The second stage is to recognise characters
###############################################

###############################################
# The first stage
###############################################
def compare(rect1, rect2):
	if abs(rect1[1] - rect2[1]) > 10:
		return rect1[1] - rect2[1]
	else:
		return rect1[0] - rect2[0]
for vax in range(94):
    # file_path=f"img/number_plates/{i}.jpg"
    
	image_1=cv2.imread(f"img/number_plates/{vax}.jpg")
	image_0=deskew(image_1)
	copied=image_0.copy()
	image_0 =(255-image_0)

	# gray=preprocessImage(image_0)
	gray = cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY)
	# # Show the original image
	# cv2.imshow("License Plate",gray)


	# Apply Gaussian blurring and thresholding 
	# to reveal the characters on the license plate
	blurred = cv2.GaussianBlur(gray, (5, 5),-1)
	#thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)
	#thresh = cv2.threshold(blurred,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
 	
	copied_1=thresh.copy()
	
 	# cv2.imshow("binary",thresh)
 	

	# Perform connected components analysis on the thresholded images and
	# initialize the mask to hold only the components we are interested in
	_, labels = cv2.connectedComponents(thresh)
	mask = np.zeros(thresh.shape, dtype="uint8")

	# Set lower bound and upper bound criteria for characters
	total_pixels = image_0.shape[0] * image_0.shape[1]
	lower = total_pixels // 110 # heuristic param, can be fine tuned if necessary
	upper = total_pixels // 20 # heuristic param, can be fine tuned if necessary

	# Loop over the unique componentsp
	for (i, label) in enumerate(np.unique(labels)):
		# If this is the background label, ignore it
		if label == 0:
			continue
	
		# Otherwise, construct the label mask to display only connected component
		# for the current label
		labelMask = np.zeros(thresh.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)
	
		# If the number of pixels in the component is between lower bound and upper bound, 
		# add it to our mask
		if numPixels > lower and numPixels < upper:
			mask = cv2.add(mask, labelMask)

	# Find contours and get bounding box for each contour
	cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]

	# Sort the bounding boxes from left to right, top to bottom
	# sort by Y first, and then sort by X if Ys are similar

	boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare) )
	# print(boundingBoxes)
	final_bbox = []
	for box in boundingBoxes:
		x,y,w,h = box 
		if (w/h < 2 and h/w< 3.5):
			final_bbox.append(box)
	boundingBoxes = final_bbox
	# Loop over the bounding boxes
	for rect in boundingBoxes:

		# Get the coordinates from the bounding box
		x,y,w,h = rect

		# Show bounding box
		cv2.rectangle(copied, (x,y), (x+w,y+h), (0, 255, 0), 2)
		cv2.rectangle(copied_1, (x,y), (x+w,y+h), (0, 255, 0), 2)

		
	# Show final image
	# cv2.imshow('Final',copied)
	cv2.imwrite(f'F:/Char_seg/img/outputs_binary/{vax}.jpg',copied_1)
	cv2.imwrite(f'F:/Char_seg/img/outputs/{vax}.jpg',copied)
	# cv2.imwrite('F:/Char_seg/img/ratoplate_binary.jpg',thresh)
	
	dirname = f'img/segment_output/{vax}'
	# os.mkdir(dirname)
	#extraction of segemnted characters
	# os.chdir(dirname)
	if os.path.exists(dirname):
		shutil.rmtree(dirname)
	os.mkdir(dirname)
	idx =0 

	for cnt in cnts:
		idx += 1
		x,y,w,h = cv2.boundingRect(cnt)
		roi=thresh[y:y+h,x:x+w]
		# gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		cv2.imwrite(os.path.join(dirname,f'{idx}.jpg'),roi)    
		# cv2.imwrite(f'{idx}.jpg',roi)
		cv2.rectangle(image_0,(x,y),(x+w,y+h),(200,0,0),2)
	cv2.waitKey(0)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	

  
