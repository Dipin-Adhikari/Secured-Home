import customtkinter
import cv2
import numpy as np
import time
from PIL import Image
class ButtonCreations(customtkinter.CTkButton):
	all_buttons_list = []
	all_on_images = []
	all_off_images = []
	all_button_state = []

	def __init__(self, master, width, height, on_image, row, column, padx, pady, off_image, function='', frame=None,
				 cap=None, mpPose=None, pose=None, mpDraw=None, facenet=None, faceDetection=None,
				 signatureDatabase=None, maskModel=None):
		self.off_image = off_image
		self.on_image = on_image

		self.function = function
		self.working_frame = frame
		self.cap = cap
		self.mpPose = mpPose
		self.pose = pose
		self.mpDraw = mpDraw
		self.facenet = facenet
		self.faceDetection = faceDetection
		self.signatureDatabase = signatureDatabase
		self.maskModel = maskModel
		self.categories = {0: 'Mask', 1: 'No Mask'}
		self.pTime, self.cTime = 0, 0

		if self.function == 'home':
			self.change_state = 1
		else:
			self.change_state = 0

		super().__init__(master=master, width=width, height=height, text="", image=self.off_image,
						 fg_color='transparent', bg_color='transparent', hover_color='#325240', command=self.change)
		self.grid(row=row, column=column, padx=padx, pady=pady)
		self.button_name = self.cget('width')
		self.all_buttons_list.append(self)
		self.all_on_images.append(self.on_image)
		self.all_off_images.append(self.off_image)
		self.all_button_state.append(self.change_state)
		self.all_buttons_list[0].configure(image=self.all_on_images[0])

	def change(self):
		if self.change_state == 0:
			self.configure(image=self.on_image)
			self.change_state = 1
		else:
			self.configure(image=self.off_image)
			self.change_state = 0

		self.execute_functions()

	def display_frames(self, img, label, w, h):
		pic = customtkinter.CTkImage(light_image=img, size=(770, 370))
		label.configure(image=pic)
		label.image = pic
		label.pack()

	def open_camera(self):
		_, frame = self.cap.read()
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = self.pose.process(rgb)
		if results.pose_landmarks:
			self.mpDraw.draw_landmarks(frame, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
			results = self.faceDetection.process(rgb)
			if results.detections:
				for id, detection in enumerate(results.detections):
					bbox_class = detection.location_data.relative_bounding_box
					h, w, c = frame.shape
					bbox = int(bbox_class.xmin * w), int(bbox_class.ymin * h), int(bbox_class.width * w), int(
						bbox_class.height * h)
					cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0))

					face = rgb[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
					# face = Image.fromarray(face)
					faceR = cv2.resize(face, (160, 160))
					faceR = np.asarray(faceR)
					faceR = np.expand_dims(faceR, axis=0)
					signature = self.facenet.embeddings(faceR)
					imgA = cv2.resize(face, (224, 224))
					imgA = np.asarray(imgA)
					nImgA = (imgA.astype(np.float32) / 127 - 1)
					data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
					data[0] = nImgA
					prediction = self.maskModel.predict(data)
					category = np.argmax(prediction[0])
					percentage = max(prediction[0]) * 100
					cv2.putText(frame, self.categories[category], (bbox[0], bbox[0]), cv2.FONT_HERSHEY_SIMPLEX, 1,
								(255, 255, 0), 2, cv2.LINE_AA)

					minDist = 100
					identity = ' '
					for key, value in self.signatureDatabase.items():
						dist = np.linalg.norm(value - signature)
						if dist < minDist:
							minDist = dist
							identity = key

					cv2.putText(frame, identity, (bbox[0] - 10, bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
								(255, 255, 0), 2, cv2.LINE_AA)
		# flipped_frame = cv2.flip(frame, 1)
		flipped_frame = frame
		self.cTime = time.time()
		fps = str(int(1 / (self.cTime - self.pTime)))
		self.pTime = self.cTime
		cv2.putText(frame, fps, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
		img = Image.fromarray(cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB))
		self.display_frames(img, self.working_frame, 500, 400)
		self.working_frame.after(20, self.open_camera)

	def execute_functions(self):
		if self.function == "home":
			self.open_camera()

		elif self.function == "camera":
			pass

		elif self.function == "logs":
			pass

		elif self.function == "send_to":
			pass

		elif self.function == "theme":
			pass

		elif self.function == "about":
			pass

		elif self.function == "settings":
			pass
