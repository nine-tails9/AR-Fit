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
	cv.imshow("dig", diff)
	cv.imshow("dig2", canvas)
	black_cnt = np.sum(canvas == 0)
	not_black = np.sum(canvas != 0) 
	# cv.imshow("dig3", img)

	person_percent = (not_black/(black_cnt + not_black)) * 100

	prev = img
	
	return person_percent

def filtr(frame):
	hsv_img = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
	cv.resize(frame,(512,512))
	mask = cv.inRange(hsv_img, low_blue, high_blue)
	res = cv.bitwise_and(frame, frame, mask = mask)
	# cv.imshow("Original",  frame)
	bw = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
	cv.imshow("Result",  bw)

	bw_black = np.sum(bw == 0)
	bw_nblack = np.sum(bw != 0)

	percent = (bw_nblack)/(bw_black + bw_nblack)
	return percent * 100

cap = cv.VideoCapture(0)
# cap = cv.VideoCapture("http://10.177.12.106:8080/?action=stream")

low_blue = np.array([0, 0, 0])
high_blue = np.array([110, 110, 110])
cnt = 0
while True:
	ret, frame = cap.read()
	frame = cv.GaussianBlur(frame, (5,5), 1)

	percent_movement = diff(frame, cnt)
	percent_filter = filtr(frame)
	if cnt == 0:
		percent_movement = 1

	cnt += 1

	print(max(percent_movement, percent_filter))
	if cv.waitKey(1) == 13:
		break

cv.waitKey(0)
cv.destroyAllWindows()
