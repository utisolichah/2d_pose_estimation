import cv2 as cv
import argparse

parser = argparse.ArgumentParser(
		description = "This program try to show two input of stere image")
parser.add_argument("--input1", help="Path to image1")
parser.add_argument("--input2", help="path to image2")

args = parser.parse_args()

input_img = [args.input1, args.input2]
window = ["show image1", "show image2"]

#def code_train(frame_param)	

def main(img, win):
	for i in range(len(img)):
		frame = cv.imread(img[i])
		frame_data = frame	
		cv.imshow(win[i], frame)

		# CODE HERE
		#code_train(frame)	

	cv.waitKey(0)

if __name__ =='__main__':	
	main(input_img, window)
	#print("hai")



