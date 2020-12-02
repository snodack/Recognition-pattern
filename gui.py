import tkinter as tk
from tkinter import Tk
from tkinter.constants import END
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import NN
import cv2

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
        self.input_panel = tk.Label(root)
        self.output_panel = tk.Label(root)
        self.tk_label = tk.Text(root, height=10, width=30)
        self.input_panel.pack()
        self.output_panel.pack()
        self.tk_label.pack()
        NN.define_model("B:\\sem7\\ro\\Recognition-pattern\\vlad_model")

    def create_widgets(self):
        self.get_path_button = tk.Button(self)
        self.get_path_button["text"] = "Выбрать файл"
        self.get_path_button["command"] = self.get_path
        self.get_path_button.pack(side="right")

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    def get_path(self):
        tk.Tk().withdraw()
        filename = askopenfilename(filetypes=[("Image files", ".jpg .png")])
        img = ImageTk.PhotoImage(Image.open(filename))
        self.open_img(filename)
        result = NN.predict_image(filename)
        self.open_output(result)
    
    def open_img(self,path):
        self.input_image = None
        img = Image.open(path)
        img = img.resize((400, 200), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        self.input_panel.configure(image = img)
        self.input_panel.image = img

    
    def open_output(self, result):
        img = Image.fromarray(result[0])
        img = img.resize((400, 200), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        self.output_panel.configure(image = img)
        self.output_panel.image = img
        self.tk_label.delete('1.0', END)
        self.tk_label.insert(tk.END, result[1] + " = " +  str(result[2]))



root = tk.Tk()
app = Application(master=root)
app.mainloop()