import cv2 as cv
import numpy as np

def bg(img):
	mask = np.zeros(img.shape[:2],np.uint8)
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	rect = (50,50,450,290)
	cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	img = img*mask2[:,:,np.newaxis]
	cv.imshow("fore", img)

prev = np.ones((512,512,3))

def diff(img, cnt):
	return 0
	global prev
	if cnt == 0:
		prev = img
		return
	diff = cv.absdiff(img, prev)

	mask = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
	th = 10
	imask = mask > th
	canvas = np.zeros_like(img, np.uint8)
	canvas[imask] = img[imask]

	cv.imshow("dig", canvas)
	# cv.imshow("dig2", canvas)
	black_cnt = np.sum(canvas == 0)
	not_black = np.sum(canvas != 0) 
	# cv.imshow("dig3", img)
	# cv.waitKey(0)
	person_percent = (not_black/(black_cnt + not_black)) * 100

	prev = img
	return person_percent

def filtr(frame, id):
	# hsv_img = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
	hsv_img = frame
	low_blue = np.array([0, 0, 0])
	high_blue = np.array([160, 160, 160])
	mask = cv.inRange(hsv_img, low_blue, high_blue)
	res = cv.bitwise_and(frame, frame, mask = mask)
	# cv.imshow("Original",  pose)
	
	pose = cv.imread("pose_" + id + ".png", 0)
	pose = cv.resize(pose, (res.shape[1], res.shape[0]))

	bw = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
	ret, bw = cv.threshold(bw, 1, 255, cv.THRESH_BINARY)
	

	res = cv.bitwise_and(bw, bw, mask = pose)
	# cv.imshow("Result", res)

	# print(pose.shape[2], res.shape[2])

	# cv.imshow("Result",  res)
	tot = np.sum(pose > -1)
	mask_black = np.sum(pose == 0)
	after_mask =  cv.countNonZero(res)

	return (tot - mask_black - after_mask)/(tot - mask_black) * 100
cnt = 0
def utility(frame, id):
	global cnt
	# frame = posemark(frame)
	# bg(frame)
	rows,cols = frame.shape[:2]

	M = cv.getRotationMatrix2D((cols/2,rows/2),90,1)
	frame = cv.warpAffine(frame,M,(cols,rows))

	percent_movement = diff(frame, cnt)
	percent_filter = filtr(frame, id)
	if cnt == 0:
		percent_movement = 1
	cnt += 1
	# print(max(percent_movement, percent_filter))
	print(percent_filter)
	cv.waitKey(0)

	return percent_filter

# cap = cv.VideoCapture("http://10.177.12.106:8080/?action=stream")
# cap = cv.VideoCapture(0)

# while True:
# 	ret, frame = cap.read()
# 	# frame = cv.resize(frame, (319, 240), interpolation = cv.INTER_AREA)
# 	# frame = posemark(frame)
# 	# bg(frame)
# 	percent_movement = diff(frame, cnt)
# 	percent_filter = filtr(frame)
# 	if cnt == 0:
# 		percent_movement = 1

# 	cnt += 1
# 	print(max(percent_movement, percent_filter))
# 	if cv.waitKey(1) == 13:
# 		break

# cv.waitKey(0)
# cv.destroyAllWindows()
