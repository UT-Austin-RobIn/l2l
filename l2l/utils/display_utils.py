import threading
import time
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from tiago_gym.utils.general_utils import run_threaded_command
import screeninfo
import cv2

class FullscreenImageDisplay:
    def __init__(self, update_interval=1, monitor_id=0):
        self.update_interval = update_interval
        monitors = screeninfo.get_monitors()
        assert monitor_id < len(monitors), "Monitor ID larger than number of connected monitors"

        self.monitor = monitors[monitor_id]
        
        run_threaded_command(self.run_mainloop)
        # self.update = False
        # run_threaded_command(self.display_image)
    
    def run_mainloop(self):
        self.root = tk.Tk()
        self.root.geometry(f'{self.monitor.width}x{self.monitor.height}+0+0')
        self.label = tk.Label(self.root)
        self.label.pack()
        self.root.bind("<Escape>", lambda e: self.root.quit())
        self.root.mainloop()
    
    def update_image(self, img):
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.label.config(image=img)
        self.label.image = img
        # self.current_image = None

    def get_resized_image(self, img):
        return cv2.resize(img, (self.monitor.width, self.monitor.height))

class FullscreenStringDisplay:
    def __init__(self, update_interval=1, monitor_id=0):
        self.update_interval = update_interval
        monitors = screeninfo.get_monitors()
        assert monitor_id < len(monitors), "Monitor ID larger than number of connected monitors"

        self.monitor = monitors[monitor_id]
        
        run_threaded_command(self.run_mainloop)
    
    def run_mainloop(self):
        self.root = tk.Tk()
        self.root.geometry(f'{self.monitor.width}x{self.monitor.height}+0+0')
        self.label = tk.Label(self.root, font=('calibri', 300, 'bold'), background='black', foreground='white')
        self.label.pack()
        self.root.bind("<Escape>", lambda e: self.root.quit())
        self.root.mainloop()
        
    def display_text(self, text, relx=None, rely=None):
        self.label.config(text=text)
        if relx is None:
            relx = np.random.random()*0.4 + 0.1
        if rely is None:
            rely = np.random.random()*0.7 + 0.1
        self.label.place(relx=relx, rely=rely)
        
# Function to generate numpy images
def generate_numpy_image(width, height):
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

# Usage example
if __name__ == "__main__":
    # display1 = FullscreenImageDisplay(update_interval=1, monitor_id=0)

    # # Simulate updates with generated numpy images
    # for x in range(3):
    #     time.sleep(1)
    #     img = np.array(Image.open("/home/shivin/Downloads/hot.png"))
    #     img = display1.get_resized_image(img)
    #     display1.update_image(img)
    # input('enter')

    display1 = FullscreenStringDisplay(update_interval=1, monitor_id=0)

    # Simulate updates with generated numpy images
    for x in range(11):
        time.sleep(1)
        display1.display_text(text="8:00 PM", rely=0.5, relx=0.3)
    input('enter')
