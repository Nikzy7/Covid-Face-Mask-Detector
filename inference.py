#importing the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2


#the necessary OpenCV, Face detection and Mask detection instnaces
cap = cv2.VideoCapture(0)
maskNet = load_model("mask_detector.model")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

'''
def predictions():
function handles the mask detection operation
receives the detected face co-ordinates (detected using OpenCV) and the frame captured
returns a tuple containing the scores for mask and without mask
'''
def predictions(face_given,frame):

	(x, y, w, h) = face_given

	face = frame[y:y+h, x:x+w]

	face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
	face = cv2.resize(face, (224, 224))
	face = img_to_array(face)
	face = preprocess_input(face)
	face = np.expand_dims(face, axis=0)

	pred = maskNet.predict(face)

	return pred
	

'''
def detect_faces():
function handles the face detection operation using OpenCV
receives the frame captured as argument
return the co-ordinates of detected face
'''
def detect_faces(frame):
	gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)

	return faces


'''
def main():
handles the main while loop and is the driver function
'''
def main():
	while True:
		
		frame = cap.read()[1]
		danger = 0
		no_risk = 0

		#scaling used to improve the FPS
		scale_percent = 80 # percent of original size
		width = int(frame.shape[1] * scale_percent / 100)
		height = int(frame.shape[0] * scale_percent / 100)
		dim = (width, height)

		h_f,w_f = frame.shape[0:2]

		frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
      
		faces = detect_faces(frame)

		outer_frame_color = (0, 255, 0)

		for face in faces:
			(x,y,w,h) = face

			#get predictions
			pred = predictions(face,frame)

			[[clear,risk]] = pred

			#generating labels
			label = "Clear !" if clear > risk else "Possible Risk !"
			color = (0, 255, 0) if label == "Clear !" else (0, 0, 255)

			if label=="Possible Risk !":
				outer_frame_color = (0, 0, 255)
				cv2.putText(frame, "Possible Threat !", (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				danger+=1
			else:
				no_risk+=1

			#handling the face rectangle and color of rectangle
			cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)

		#handling the frame rectangle and other statistical info
		cv2.rectangle(frame,(0,0),(w_f-130,h_f-100),outer_frame_color,6)
		cv2.putText(frame, "Detected Faces : "+str(len(faces)), (10,18),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,222), 2)
		cv2.putText(frame, "Positive Threat Count : "+str(danger), (10,31),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		cv2.putText(frame, "Negative Threat Count : "+str(no_risk), (10,44),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

		cv2.imshow("Face Mask Detector - BETA Jetson Nano", frame)

		#press q to quit
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if __name__ == "__main__":
	main()