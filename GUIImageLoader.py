import customtkinter
from PIL import Image

class ImageLoader(customtkinter.CTkImage):

	def __init__(self, file_path, size=(20, 20)):
		super().__init__(light_image=Image.open(file_path), size=size)
