import tkinter as tk
from tkinter import Tk
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
        NN.define_model("B:\\sem7\\ro\\Recognition-pattern\\ultra_last_model.h5")
        result = NN.predict_image(filename)
        self.open_output(result)
    
    def open_img(self,path):
        img = Image.open(path)
        img = img.resize((400, 200), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = tk.Label(root, image=img)
        panel.image = img
        panel.pack()
    
    def open_output(self, result):
        image = Image.fromarray(result[0])
        image = image.resize((400, 200), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image)
        panel = tk.Label(root, image=img)
        panel.output_image = img
        panel.pack()
        T = tk.Text(root, height=10, width=30)
        T.insert(tk.END, result[1] + " = " +  str(result[2]))
        T.pack()



root = tk.Tk()
app = Application(master=root)
app.mainloop()