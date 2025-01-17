import serial
import time
from constants import RobotConstants


class Robot(object):
    def __init__(
        self, port: str = "COM4", baudrate: int = RobotConstants.BAUDRATE, debug: bool = False
    ) -> None:
        if debug:
            print("Running in DEBUG mode")
            self._serial_connection = None
        else:
            self._serial_connection = serial.Serial()
            self._serial_connection.port = port
            self._serial_connection.baudrate = baudrate

    def home(self):
        self.send_gcode("G28G29")
        time.sleep(RobotConstants.HOME_TIME)

    def send_gcode(self, command: str) -> str:
        if self._serial_connection:
            buffer = self._serial_connection.read_all()
            if b"M999" in buffer:
                print("Trying again due to failure...")
                self.home()
                return self.send_gcode(command)
            self._serial_connection.write((command + "\n").encode())
            time.sleep(RobotConstants.COMMAND_TIME)
            response = self._serial_connection.readline().decode().strip()
            # print("Serial RX:", response)
            return response
        else:
            time.sleep(RobotConstants.COMMAND_TIME)
            return command

    def translate_x(self, distance: float, speed: float = RobotConstants.FEED_RATE) -> str:
        g_code = f"G0 X{distance:.2f} F{speed:.2f}"
        response = self.send_gcode(g_code)
        print(response)
        return response

    def translate_y(self, distance: float, speed: float = RobotConstants.FEED_RATE) -> str:
        g_code = f"G0 Y{distance:.2f} F{speed:.2f}"
        response = self.send_gcode(g_code)
        print(response)
        return response

    def translate_z(self, distance: float, speed: float = RobotConstants.FEED_RATE) -> str:
        g_code = f"G0 Z{distance:.2f} F{speed:.2f}"
        response = self.send_gcode(g_code)
        print(response)
        return response

    def go_to(self, x_position, y_position, z_position) -> str:
        # print(x_position, y_position, z_position)
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
        if self._serial_connection:
            self._serial_connection.open()
            response_1 = self.send_gcode(RobotConstants.UNITS)
            if response_1.lower() == "ok":
                return "ok"

            raise IOError(
                "Error during robot initialization; unrecognized command.")

        else:
            print("DEBUG MODE: start")

    def end(self) -> None:
        if self._serial_connection:
            self._serial_connection.close()
        else:
            print("DEBUG MODE: end")
