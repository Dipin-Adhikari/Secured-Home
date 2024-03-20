import random
import time
import mediapipe as mp
import customtkinter
from PIL import Image
import cv2
import numpy as np
from keras_facenet import FaceNet
import pickle
import os
from tensorflow.keras.models import load_model


class SwitchCreations(customtkinter.CTkSwitch):
	def __init__(self, master, width, height, switch_width, switch_height, corner_radius, border_width, fg_color, progress_color, button_color, row, column,state='disabled'):
		super().__init__(master=master, width=width, height=height, text="",switch_width=switch_width, switch_height=switch_height, corner_radius=corner_radius, border_width=border_width, fg_color=fg_color, progress_color=progress_color, button_color=button_color)
		self.grid(row=row, column=column, sticky='e')
		if random.randint(0,1) == 1:
			self.select()
		self.configure(state=state)


class LabelCreations(customtkinter.CTkLabel):

	def __init__(self, master, text, row, column, padx, pady, image=None, width=200, height=30, sticky='w',
				 columnspan=1):
		super().__init__(master=master, width=width, height=height, text=text, text_color="#bdbdbd",
						 font=("Minado Rough Demo", 24), justify='left', anchor='w', fg_color='transparent',
						 image=image)
		self.grid(row=row, column=column, padx=15, pady=15, sticky=sticky, columnspan=columnspan)


class ButtonCreations(customtkinter.CTkButton):
	all_buttons_list = []
	all_on_images = []
	all_off_images = []
	all_button_state = []

	def __init__(self, master, width, height, on_image, row, column, padx, pady, off_image, function='', frame=None, cap=None, mpPose=None, pose=None, mpDraw=None, facenet=None, faceDetection=None, signatureDatabase=None, maskModel=None):
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
		self.pTime, self.cTime = 0,0

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
						bbox = int(bbox_class.xmin * w), int(bbox_class.ymin * h), int(bbox_class.width * w), int(bbox_class.height * h)
						cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0))

						face = rgb[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
						# face = Image.fromarray(face)
						faceR = cv2.resize(face, (160, 160))
						faceR = np.asarray(faceR)
						faceR = np.expand_dims(faceR, axis=0)
						signature = self.facenet.embeddings(faceR)
						imgA = cv2.resize(face, (224, 224))
						imgA = np.asarray(imgA)
						nImgA = (imgA.astype(np.float32) / 127 -1)
						data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
						data[0] = nImgA
						prediction = self.maskModel.predict(data)
						category = np.argmax(prediction[0])
						percentage = max(prediction[0]) * 100
						cv2.putText(frame, self.categories[category], (bbox[0], bbox[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

						minDist=100
						identity=' '
						for key, value in self.signatureDatabase.items() :
							dist = np.linalg.norm(value-signature)
							if dist < minDist:
								minDist = dist
								identity = key
						
						cv2.putText(frame,identity, (bbox[0]-10, bbox[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
		# flipped_frame = cv2.flip(frame, 1)
		flipped_frame = frame
		self.cTime = time.time()
		fps = str(int(1 / (self.cTime-self.pTime)))
		self.pTime = self.cTime
		cv2.putText(frame,fps, (200,200),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
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


class FrameCreations(customtkinter.CTkFrame):

	def __init__(self, master, width, height, fg_color, bg_color, corner_radius, row, column, padx=0, pady=0, rowspan=1,
				 columnspan=1):
		super().__init__(master, width=width, height=height, fg_color=fg_color, bg_color=bg_color,
						 corner_radius=corner_radius)
		self.grid(row=row, column=column, padx=padx, pady=pady, rowspan=rowspan, columnspan=columnspan, sticky='nsew')


class ImageLoader(customtkinter.CTkImage):

	def __init__(self, file_path, size=(20, 20)):
		super().__init__(light_image=Image.open(file_path), size=size)


class MyGUI(customtkinter.CTk):

	def __init__(self):
		super().__init__()

		self.title("Thief Detection")
		self.path = r"C:\User Files\Projects\GUI_DETECTOR\IMAGES\ "
		self.cap = cv2.VideoCapture(0)
		# self.cap = cv2.VideoCapture(r"D:\Secured Home\SecuredHome\Project Files\Video\robbery2.mp4")
		self.facenet = FaceNet()
		self.faceDetector = mp.solutions.face_detection
		self.faceDetection = self.faceDetector.FaceDetection()
		self.mpPose = mp.solutions.pose
		self.pose = self.mpPose.Pose(min_detection_confidence=0.7)
		self.mpDraw = mp.solutions.drawing_utils
		self.signatureDatabase = pickle.load(open("signatureData.pkl", "rb"))
		self.maskModel = load_model("D:\Secured Home\SecuredHome\Project Files\Face Mask Detection Model.h5")

		self.load_images()
		self.create_frames()
		self.create_labels()
		self.create_buttons()
		self.create_switches()
		self.row_column_configure()
		self.extract_data()
		


	def load_images(self):
		self.home_off = ImageLoader(r'D:\Secured Home\SecuredHome\Project Files\GUI_DETECTOR\Images\home_off.png')
		self.home_on = ImageLoader(r'D:\Secured Home\SecuredHome\Project Files\GUI_DETECTOR\Images\home_on.png')
		self.send_off = ImageLoader(r'D:\Secured Home\SecuredHome\Project Files\GUI_DETECTOR\Images\send_off.png')
		self.send_on = ImageLoader(r'D:\Secured Home\SecuredHome\Project Files\GUI_DETECTOR\Images\send_on.png')
		self.logs_off = ImageLoader(r'D:\Secured Home\SecuredHome\Project Files\GUI_DETECTOR\Images\logs_off.png')
		self.logs_on = ImageLoader(r'D:\Secured Home\SecuredHome\Project Files\GUI_DETECTOR\Images\logs_on.png')
		self.camera_on = ImageLoader(r'D:\Secured Home\SecuredHome\Project Files\GUI_DETECTOR\Images\camera_on.png')
		self.camera_off = ImageLoader(r'D:\Secured Home\SecuredHome\Project Files\GUI_DETECTOR\Images\camera_off.png')
		self.theme_off = ImageLoader(r'D:\Secured Home\SecuredHome\Project Files\GUI_DETECTOR\Images\theme_off.png')
		self.theme_on = ImageLoader(r'D:\Secured Home\SecuredHome\Project Files\GUI_DETECTOR\Images\theme_on.png')
		self.settings_off = ImageLoader(r'D:\Secured Home\SecuredHome\Project Files\GUI_DETECTOR\Images\settings_off.png')
		self.settings_on = ImageLoader(r'D:\Secured Home\SecuredHome\Project Files\GUI_DETECTOR\Images\settings_on.png')
		self.about_on = ImageLoader(r'D:\Secured Home\SecuredHome\Project Files\GUI_DETECTOR\Images\about_on.png')
		self.about_off = ImageLoader(r'D:\Secured Home\SecuredHome\Project Files\GUI_DETECTOR\Images\about_off.png')

		self.person = ImageLoader(r'D:\Secured Home\SecuredHome\Project Files\GUI_DETECTOR\Images\person.png', size=(40, 40))
		self.mask = ImageLoader(r'D:\Secured Home\SecuredHome\Project Files\GUI_DETECTOR\Images\wearing-mask.png', size=(40, 40))
		self.weapon = ImageLoader(r'D:\Secured Home\SecuredHome\Project Files\GUI_DETECTOR\Images\weapon.png', size=(40, 40))
		self.weather = ImageLoader(r'D:\Secured Home\SecuredHome\Project Files\GUI_DETECTOR\Images\weather.png', size=(40, 40))

	def create_frames(self):
		self.menubarframe = FrameCreations(self, 50, 768, "#5F8670", "#252525", 10, 0, 0, rowspan=4)
		self.header = FrameCreations(self, 975, 50, "#242424", "#252525", 10, 0, 1, (0, 10), (0, 10), 1, 3)
		self.cameraframe = FrameCreations(self, 800, 400, '#242424', '#252525', 30, 1, 1, (18, 20), (10, 10),
										  columnspan=2)
		self.detail_frame1 = FrameCreations(self, 390, 120, "#393E46", "#252525", 30, 2, 1, (18, 10), (15, 15))
		self.detail_frame2 = FrameCreations(self, 390, 120, "#393E46", "#252525", 30, 2, 2, (10, 10), (15, 15))
		self.detail_frame3 = FrameCreations(self, 390, 120, "#393E46", "#252525", 30, 3, 1, (18, 10), (0, 15))
		self.detail_frame4 = FrameCreations(self, 390, 120, "#393E46", "#252525", 30, 3, 2, (10, 10), (0, 15))

	def create_buttons(self):
		self.home_button = ButtonCreations(self.menubarframe, 40, 40, self.home_on, 0, 0, 5, (40, 10), self.home_off,
										   'home', frame=self.camera_label,cap=self.cap, mpPose=self.mpPose, pose=self.pose, mpDraw=self.mpDraw, facenet=self.facenet, faceDetection=self.faceDetection, signatureDatabase=self.signatureDatabase, maskModel=self.maskModel)
		self.full_screen_camera_button = ButtonCreations(self.menubarframe, 40, 40, self.camera_on, 1, 0, 5, (40, 10),
														 self.camera_off)
		self.share_to_button = ButtonCreations(self.menubarframe, 40, 40, self.send_on, 2, 0, 5, (40, 10),
											   self.send_off)
		self.logs_history_button = ButtonCreations(self.menubarframe, 40, 40, self.logs_on, 3, 0, 5, (40, 10),
												   self.logs_off)

		self.theme_button = ButtonCreations(self.menubarframe, 40, 40, self.theme_on, 4, 0, 5, (40, 10),
											self.theme_off)
		self.about_button = ButtonCreations(self.menubarframe, 40, 40, self.about_on, 5, 0, 5, (200, 10),
											self.about_off)
		self.settings_button = ButtonCreations(self.menubarframe, 40, 40, self.settings_on, 6, 0, 5, (40, 10),
											   self.settings_off)

	def create_labels(self):

		self.header_text = LabelCreations(self.header, text="Welcome to our Thief Detection System", row=0, column=0, padx=10, pady=10)

		self.camera_label = LabelCreations(self.cameraframe, text=" ", width=770, height=370, row=0, column=0,
										   padx=10, pady=10, )
		self.person_detection_text = LabelCreations(self.detail_frame1, "Person Identified", 0, 0, (0, 10), 10)
		self.person_detection_image = LabelCreations(self.detail_frame1, "", 0, 1, 0, 10, self.person, width=40,
													 height=40, sticky='e')
		self.person_detection_text_result = LabelCreations(self.detail_frame1, "False", 1, 0, 10, 10)

		self.mask_on_off_text = LabelCreations(self.detail_frame2, "Mask Identified", 0, 0, 10, 10)
		self.mask_image = LabelCreations(self.detail_frame2, "", 0, 1, 0, 10, self.mask, width=40, height=40,
										 sticky='e')
		self.mask_on_off_text_result = LabelCreations(self.detail_frame2, "False", 1, 0, 10, 10)

		self.weapon_detection_text = LabelCreations(self.detail_frame3, "Weapon Detected", 0, 0, 10, 10)
		self.weapon_image = LabelCreations(self.detail_frame3, "", 0, 1, 0, 10, self.weapon, width=40, height=40,
										   sticky='e')
		self.weapon_detection_text_result = LabelCreations(self.detail_frame3, "True", 1, 0, 10, 10)

		self.weather_conditions_text = LabelCreations(self.detail_frame4, "Weather Conditon", 0, 0, 10, 10)
		self.weather_image = LabelCreations(self.detail_frame4, "", 0, 1, 0, 10, self.weather, width=40, height=40,
											sticky='e')
		self.weather_conditions_text_result = LabelCreations(self.detail_frame4, "Foggy", 1, 0, 10, 10)

	def row_column_configure(self):
		self.columnconfigure(0,weight=0)
		self.columnconfigure(1,weight=1)
		self.columnconfigure(2,weight=1)

		self.detail_frame1.rowconfigure(0, weight=1)
		self.detail_frame1.columnconfigure(0, weight=1)
		self.detail_frame1.columnconfigure(1, weight=1)

		self.detail_frame2.rowconfigure(0, weight=1)
		self.detail_frame2.columnconfigure(0, weight=1)
		self.detail_frame2.columnconfigure(1, weight=1)

		self.detail_frame3.rowconfigure(0, weight=1)
		self.detail_frame3.columnconfigure(0, weight=1)
		self.detail_frame3.columnconfigure(1, weight=1)

		self.detail_frame4.rowconfigure(0, weight=1)
		self.detail_frame4.columnconfigure(0, weight=1)
		self.detail_frame4.columnconfigure(1, weight=1)

	def create_switches(self):
		self.person_detection_switch = SwitchCreations(self.detail_frame1, 100, 40, 60, 30, 60, 0, '#8a8a8a', "#8c52ff", '#ffffff',1,1)
		self.person_detection_switch = SwitchCreations(self.detail_frame2, 100, 40, 60, 30, 60, 0, '#8a8a8a', "#8c52ff", '#ffffff',1,1)
		self.person_detection_switch = SwitchCreations(self.detail_frame3, 100, 40, 60, 30, 60, 0, '#8a8a8a', "#8c52ff", '#ffffff',1,1)
		self.person_detection_switch = SwitchCreations(self.detail_frame4, 100, 40, 60, 30, 60, 0, '#8a8a8a', "#8c52ff", '#ffffff',1,1)

	def extract_data(self):
		self.is_person_identified = False
		self.is_mask_on = False
		self.is_weapon_identified = False

app = MyGUI()
app.mainloop()
