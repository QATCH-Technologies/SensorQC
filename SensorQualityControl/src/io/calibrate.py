import json
import tkinter as tk
from tkinter import messagebox
from tkinter import Label
import serial
import time
import cv2
import numpy as np
from threading import Thread
from PIL import Image, ImageTk

# Replace with your G-code machine's COM port and baud rate
ser = serial.Serial()
ser.port = "COM3"
ser.baudrate = 9600
time.sleep(2)  # Wait for connection to establish

# Initial positions
current_x = 0
current_y = 0
current_z = 0

# Initialize main window
root = tk.Tk()
root.title("Calibration GUI")

# Camera feed label
camera_label = Label(root)
camera_label.grid(row=0, column=0, columnspan=4)


# Function to send G-code commands
def send_gcode(command):
    ser.write((command + "\n").encode())
    time.sleep(0.1)


# Movement functions
def move_x(distance):
    global current_x
    g_code = f"G0 X{distance}"
    send_gcode(g_code)
    current_x += distance


def move_y(distance):
    global current_y
    g_code = f"G0 Y{distance}"
    send_gcode(g_code)
    current_y += distance


def move_z(distance):
    global current_z
    g_code = f"G0 Z{distance}"
    send_gcode(g_code)
    current_z += distance


# Initialize machine parameters
def init_params():
    units_selection = "G21"  # Set units to millimeters
    send_gcode(units_selection)
    positioning_relative = "G91"  # Set to relative positioning
    send_gcode(positioning_relative)
    xy_plane = "G17"  # Select XY plane
    send_gcode(xy_plane)


# Move machine to initial position
def move_to_initial_position():
    move_x(100)
    move_y(100)


# Log the current position
def log_position():
    position = {"X": current_x, "Y": current_y, "Z": current_z}
    print(f"Logging Position - X: {current_x}, Y: {current_y}, Z: {current_z}")
    with open("logged_positions.json", "a") as log_file:
        json.dump(position, log_file)
        log_file.write("\n")


# Camera feed function
def display_camera_feed():
    cap = cv2.VideoCapture(0)  # Default camera
    if not cap.isOpened():
        messagebox.showinfo("Camera Feed", "No camera connected.")
        camera_label.config(text="No Camera Feed")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Crosshairs
        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2
        cv2.line(
            frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2
        )
        cv2.line(
            frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 2
        )

        # Convert frame to display in tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update GUI label with the camera feed
        camera_label.imgtk = imgtk
        camera_label.config(image=imgtk)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    camera_label.config(image="")


# Thread for camera feed
camera_thread = Thread(target=display_camera_feed)
camera_thread.start()

# Control buttons
move_up_button = tk.Button(root, text="Move Up", command=lambda: move_y(1))
move_up_button.grid(row=1, column=1)
move_down_button = tk.Button(root, text="Move Down", command=lambda: move_y(-1))
move_down_button.grid(row=3, column=1)
move_left_button = tk.Button(root, text="Move Left", command=lambda: move_x(-1))
move_left_button.grid(row=2, column=0)
move_right_button = tk.Button(root, text="Move Right", command=lambda: move_x(1))
move_right_button.grid(row=2, column=2)

move_up_z_button = tk.Button(root, text="Move Z Up", command=lambda: move_z(1))
move_up_z_button.grid(row=1, column=3)
move_down_z_button = tk.Button(root, text="Move Z Down", command=lambda: move_z(-1))
move_down_z_button.grid(row=3, column=3)

log_button = tk.Button(root, text="Log Position", command=log_position)
log_button.grid(row=4, column=1, columnspan=2)

# Initialize and start
ser.open()
init_params()
move_to_initial_position()

root.mainloop()

# Ensure the camera thread finishes
camera_thread.join()

ser.close()
