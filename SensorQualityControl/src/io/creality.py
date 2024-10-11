import keyboard
import serial
import time
import cv2
from pynput.mouse import Listener  # Use pynput for scroll detection
from datetime import datetime


BAUDRATE = 115200
COMMAND_TIME = 1
QUERY_TIME = 0.5
UNITS = "G21"
MODE = "G91"


class Robot:
    def __init__(self, port: str = 'COM4') -> None:
        self.__serial__ = serial.Serial(port, BAUDRATE)
        time.sleep(COMMAND_TIME)

    def send_gcode(self, command: str) -> str:
        self.__serial__.write((command + "\n").encode())
        response = self.__serial__.readline().decode().strip()
        return response

    def translate_x(self, distance: float, speed: float) -> str:
        g_code = f"G0 X{distance} F{speed}"
        response = self.send_gcode(g_code)
        return response

    def translate_y(self, distance: float, speed: float) -> str:
        g_code = f"G0 Y{distance} F{speed}"
        response = self.send_gcode(g_code)
        return response

    def translate_z(self, distance: float, speed: float) -> str:
        g_code = f"G0 Z{distance} F{speed}"
        response = self.send_gcode(g_code)
        return response

    def get_absolute_position(self):
        g_code = "M114"
        response = self.send_gcode(g_code)
        time.sleep(QUERY_TIME)
        return response

    def begin(self):
        self.__serial__.open()
        self.send_gcode(UNITS)
        self.send_gcode(MODE)

    def end(self):
        self.__serial__.close()


class Controls:
    def __init__(self, robot: Robot):
        self.__robot__ = robot
        self.__robot__.begin()
        self.last_scroll_time = None

    def on_scroll(self, scroll_event):
        time_threshold = 0.1
        base_zoom_step = 0.1
        max_zoom_step = 1.0
        current_time = time.time()

        # Calculate time difference since last scroll
        if self.last_scroll_time is None:
            time_difference = time_threshold
        else:
            time_difference = current_time - self.last_scroll_time

        # Store the time of the current scroll
        self.last_scroll_time = current_time

        # Inverse relation between time difference and zoom step: faster scroll -> bigger zoom step
        if time_difference < time_threshold:
            zoom_step = max_zoom_step
        else:
            zoom_step = base_zoom_step + \
                (max_zoom_step - base_zoom_step) * \
                (time_threshold / time_difference)

        # Apply zoom direction based on scroll delta
        if scroll_event.delta > 0:
            self.zoom_in(zoom_step)
        else:
            self.zoom_out(zoom_step)

    def zoom_out(self, zoom_step):
        self.__robot__.translate_z(zoom_step)
        print(f"Zooming out by {zoom_step}")

    def zoom_in(self, zoom_step):
        self.__robot__.translate_z(zoom_step)
        print(f"Zooming in by {zoom_step}")


if __name__ == '__main__':
    r = Robot('COM1')
    c = Controls(r)
    r.end()
