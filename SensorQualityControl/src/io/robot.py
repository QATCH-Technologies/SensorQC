import serial
import time

BAUDRATE = 115200
COMMAND_TIME = 0.5
FEED_RATE = 400
UNITS = "G21"
MODE = "G91"


class Robot:
    def __init__(self, port: str = 'COM4', baudrate: int = BAUDRATE, debug: bool = False) -> None:
        if debug:
            print("Running in DEBUG mode")
            self.__serial__ = None
        else:
            self.__serial__ = serial.Serial()
            self.__serial__.port = port
            self.__serial__.baudrate = baudrate

    def send_gcode(self, command: str) -> str:
        if self.__serial__:
            self.__serial__.write((command + "\n").encode())
            time.sleep(COMMAND_TIME)
            response = self.__serial__.readline().decode().strip()
            return response
        else:
            return command

    def translate_x(self, distance: float, speed: float = FEED_RATE) -> str:
        g_code = f"G0 X{distance} F{speed}"
        response = self.send_gcode(g_code)
        return response

    def translate_y(self, distance: float, speed: float = FEED_RATE) -> str:
        g_code = f"G0 Y{distance} F{speed}"
        response = self.send_gcode(g_code)
        return response

    def translate_z(self, distance: float, speed: float = FEED_RATE) -> str:
        g_code = f"G0 Z{distance} F{speed}"
        response = self.send_gcode(g_code)
        return response

    def get_absolute_position(self) -> str:
        g_code = "M114"
        response = self.send_gcode(g_code)
        print(response)
        return response

    def begin(self) -> str:
        if self.__serial__:
            self.__serial__.open()
            response_1 = self.send_gcode(UNITS)
            response_2 = self.send_gcode(MODE)
            if response_1.lower() == 'ok' and response_2.lower() == 'ok':
                return 'ok'
            raise IOError(
                'Error during robot initialization; unrecognized command.')
        else:
            print('DEBUG MODE: start')

    def end(self) -> None:
        if self.__serial__:
            self.__serial__.close()
        else:
            print('DEBUG MODE: end')
