from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

filename = askopenfilename(filetypes = [("txt files", "*.txt"), ("excel files", "*.csv"), ("excel files", "*.xlsx")])

f = open(filename, "r")

print(filename)

