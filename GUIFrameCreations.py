import customtkinter

class FrameCreations(customtkinter.CTkFrame):

	def __init__(self, master, width, height, fg_color, bg_color, corner_radius, row, column, padx=0, pady=0, rowspan=1,
				 columnspan=1):
		super().__init__(master, width=width, height=height, fg_color=fg_color, bg_color=bg_color,
						 corner_radius=corner_radius)
		self.grid(row=row, column=column, padx=padx, pady=pady, rowspan=rowspan, columnspan=columnspan, sticky='nsew')
