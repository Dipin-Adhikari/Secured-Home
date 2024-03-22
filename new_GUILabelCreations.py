import customtkinter

class LabelCreations(customtkinter.CTkLabel):

	def __init__(self, master, text, row, column, padx, pady, image=None, width=200, height=30, sticky='w',
				 columnspan=1):
		super().__init__(master=master, width=width, height=height, text=text, text_color="#bdbdbd",
						 font=("Minado Rough Demo", 24), justify='left', anchor='w', fg_color='transparent',
						 image=image)
		self.grid(row=row, column=column, padx=15, pady=15, sticky=sticky, columnspan=columnspan)

