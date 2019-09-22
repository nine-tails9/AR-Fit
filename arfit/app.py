from flask import Flask, jsonify
from flask import request
import cv2 as cv
import base64
import sys
sys.path.append('../')
from body import utility
app = Flask(__name__)

@app.route("/", methods = ['POST'])
def hello():
	data = request.form['snap']
	poseId = request.form['id']
	imgdata = base64.b64decode(data)
	filename = 'test.jpg'  # I assume you have a way of picking unique filenames
	with open(filename, 'wb') as f:
		f.write(imgdata)
	img = cv.imread("test.jpg")
	score = str(utility(img, poseId))
	print(poseId)

	return score, 200


cv.waitKey()
cv.destroyAllWindows()
if __name__ == "__main__":
  app.run()



