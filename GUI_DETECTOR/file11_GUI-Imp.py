import random
import time
import random

import customtkinter
from PIL import Image
import cv2
import numpy as np 
import os 
from keras_facenet import FaceNet
import pickle
import mediapipe as mp

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

	def __init__(self, master, width, height, on_image, row, column, padx, pady, off_image, function='', frame=None):
		self.off_image = off_image
		self.on_image = on_image

		self.function = function
		self.working_frame = frame
		if self.function == 'home':
			self.change_state = 1
		else:
			self.change_state = 0

		super().__init__(master=master, width=width, height=height, text="", image=self.off_image,
						 fg_color='transparent', bg_color='transparent', hover_color='#325240', command=self.change)
		self.grid(row=row, column=column, padx=padx, pady=pady)
		self.button_name = self.cget('width')
		print(self.button_name)
		self.all_buttons_list.append(self)
		self.all_on_images.append(self.on_image)
		self.all_off_images.append(self.off_image)
		self.all_button_state.append(self.change_state)
		print(self.all_buttons_list)
		self.all_buttons_list[0].configure(image=self.all_on_images[0])

	# self.all_button_state[0] = 1
	# for _ in self.all_buttons_list:
	#     print(_.configure(text = "HaagagR"))

	def change(self):
		print(self.all_button_state)
		print(self)
		print(self.change_state)
		if self.change_state == 0:
			print('hello')
			self.configure(image=self.on_image)
			self.change_state = 1
		else:
			print('hello')
			self.configure(image=self.off_image)
			self.change_state = 0

		self.execute_functions()

	def open_camera(self):
		print("Camera Avoked")
		cap = cv2.VideoCapture(r"D:\Secured Home\SecuredHome\Project Files\Video\robbery2.mp4")
				
		facenet = FaceNet()
		faceDetector = mp.solutions.face_detection   
		faceDetection = faceDetector.FaceDetection()
		mpPose = mp.solutions.pose
		pose = mpPose.Pose(min_detection_confidence=0.7)
		mpDraw = mp.solutions.drawing_utils

		myFile = open("signatureData.pkl", "rb")
		signatureDatabase = pickle.load(myFile)
		myFile.close()

		cTime, pTime = 0, 0

		while True:
			ret, img = cap.read()
			# print(ret)
			# print(frame)
			if not ret:
				break

			imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			results = pose.process(imgRgb)
			if results.pose_landmarks:
				mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
				results = faceDetection.process(imgRgb)
				if results.detections:
						for id, detection in enumerate(results.detections):
							bbox_class = detection.location_data.relative_bounding_box
							h, w, c = img.shape
							bbox = int(bbox_class.xmin * w), int(bbox_class.ymin * h), int(bbox_class.width * w), int(bbox_class.height * h)
							cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0))
					# else:
					#     bbox = [1, 1, 10, 10]

							face = imgRgb[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
							face = Image.fromarray(face)
							face = face.resize((160, 160))
							face = np.asarray(face)
							face = np.expand_dims(face, axis=0)
							signature = facenet.embeddings(face)

							minDist=100
							identity=' '
							for key, value in signatureDatabase.items() :
								dist = np.linalg.norm(value-signature)
								if dist < minDist:
									minDist = dist
									identity = key
							
							cv2.putText(img,identity, (bbox[0]-10, bbox[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
			cTime = time.time()
			fps = str(int(1 / (cTime-pTime)))
			pTime = cTime
			cv2.putText(img,fps, (200,200),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

			flippedImg = cv2.flip(img, 1)
			imgN = Image.fromarray(cv2.cvtColor(flippedImg, cv2.COLOR_BGR2RGB))

			new_image = customtkinter.CTkImage(light_image=imgN, size=(770, 370))
			self.working_frame.configure(image=new_image)
			self.working_frame.update_idletasks()

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

# elif self.function ==


class FrameCreations(customtkinter.CTkFrame):

	def __init__(self, master, width, height, fg_color, bg_color, corner_radius, row, column, padx=0, pady=0, rowspan=1,
				 columnspan=1):
		super().__init__(master, width=width, height=height, fg_color=fg_color, bg_color=bg_color,
						 corner_radius=corner_radius)
		self.grid(row=row, column=column, padx=padx, pady=pady, rowspan=rowspan, columnspan=columnspan, sticky='nsew')
		print(type(self))


class ImageLoader(customtkinter.CTkImage):

	def __init__(self, file_path, size=(20, 20)):
		super().__init__(light_image=Image.open(file_path), size=size)
		print(self.cget('size'))


class MyGUI(customtkinter.CTk):

	def __init__(self):
		super().__init__()

		self.title("Thief Detection")

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
										   'home', frame=self.camera_label)
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

		self.camera_label = LabelCreations(self.cameraframe, text="", width=770, height=370, row=0, column=0,
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
		# self.columnconfigure(0,weight=0)

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
		self.is_person_identified = True
		self.is_mask_on = True
		self.is_weapon_identified = True

app = MyGUI()
app.mainloop()
