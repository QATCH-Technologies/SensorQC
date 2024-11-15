import serial
import time

BAUDRATE = 115200
COMMAND_TIME = 0.5
FEED_RATE = 1000
UNITS = "G21"
HOME_TIME = 20


class Robot(object):
    def __init__(
        self, port: str = "COM4", baudrate: int = BAUDRATE, debug: bool = False
    ) -> None:
        if debug:
            print("Running in DEBUG mode")
            self.__serial__ = None
        else:
            self.__serial__ = serial.Serial()
            self.__serial__.port = port
            self.__serial__.baudrate = baudrate

    def home(self):
        self.send_gcode("G28G29")
        time.sleep(HOME_TIME)

    def send_gcode(self, command: str) -> str:
        if self.__serial__:
            buffer = self.__serial__.read_all()
            if b"M999" in buffer:
                print("Trying again due to failure...")
                self.home()
                return self.send_gcode(command)
            self.__serial__.write((command + "\n").encode())
            time.sleep(COMMAND_TIME)
            response = self.__serial__.readline().decode().strip()
            print("Serial RX:", response)
            return response
        else:
            time.sleep(COMMAND_TIME)
            return command

    def translate_x(self, distance: float, speed: float = FEED_RATE) -> str:
        g_code = f"G0 X{distance:.2f} F{speed:.2f}"
        response = self.send_gcode(g_code)
        print(response)
        return response

    def translate_y(self, distance: float, speed: float = FEED_RATE) -> str:
        g_code = f"G0 Y{distance:.2f} F{speed:.2f}"
        response = self.send_gcode(g_code)
        print(response)
        return response

    def translate_z(self, distance: float, speed: float = FEED_RATE) -> str:
        g_code = f"G0 Z{distance:.2f} F{speed:.2f}"
        response = self.send_gcode(g_code)
        print(response)
        return response

    def go_to(self, x_position, y_position, z_position) -> str:
        print(x_position, y_position, z_position)
        g_code = f"G00 X{x_position:.2f} Y{y_position:.2f} Z{z_position:.2f}\n"
        response = self.send_gcode(g_code)
        return response

    def absolute_mode(self) -> str:
        print("[INFO] Running in absolute mode.")
        g_code = "G90"
        response = self.send_gcode(g_code)
        return response

    def relative_mode(self) -> str:
        print("[INFO] Running in relative mode.")
        g_code = "G91"
        response = self.send_gcode(g_code)
        return response

    def get_absolute_position(self) -> str:
        mode_switch = self.absolute_mode()
        print(mode_switch)
        response = None
        if mode_switch.lower() == "ok":
            g_code = "M114"
            response = self.send_gcode(g_code)
            print(response)
        self.relative_mode()
        return response

    def begin(self) -> str:
        if self.__serial__:
            self.__serial__.open()
            response_1 = self.send_gcode(UNITS)
            if response_1.lower() == "ok":
                return "ok"

            raise IOError("Error during robot initialization; unrecognized command.")

        else:
            print("DEBUG MODE: start")

    def end(self) -> None:
        if self.__serial__:
            self.__serial__.close()
        else:
            print("DEBUG MODE: end")
