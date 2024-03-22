import customtkinter
import random

class SwitchCreations(customtkinter.CTkSwitch):
	def __init__(self, master, width, height, switch_width, switch_height, corner_radius, border_width, fg_color,
				 progress_color, button_color, row, column, state='disabled'):
		super().__init__(master=master, width=width, height=height, text="", switch_width=switch_width,
						 switch_height=switch_height, corner_radius=corner_radius, border_width=border_width,
						 fg_color=fg_color, progress_color=progress_color, button_color=button_color)
		self.grid(row=row, column=column, sticky='e')
		if random.randint(0, 1) == 1:
			self.select()
		self.configure(state=state)

