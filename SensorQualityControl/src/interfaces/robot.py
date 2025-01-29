import serial
import time
import logging
from constants import RobotConstants, SystemConstants

logger = logging.getLogger(__name__)


class Robot(object):
    def __init__(
        self,
        port: str = RobotConstants.ROBOT_PORT,
        baudrate: int = RobotConstants.BAUDRATE,
        debug: bool = SystemConstants.DEBUG,
    ) -> None:
        if debug:
            logger.debug("Running in DEBUG mode")
            self._serial = None
        else:
            self._serial = serial.Serial()
            self._serial.port = port
            self._serial.baudrate = baudrate

    def home(self):
        self.send_gcode("G28G29")
        time.sleep(RobotConstants.HOMING_TIME)

    def send_gcode(self, command: str) -> str:
        if self._serial:
            buffer = self._serial.read_all()
            if b"M999" in buffer:
                logger.warning("Trying again due to failure...")
                self.home()
                return self.send_gcode(command)
            self._serial.write((command + "\n").encode())
            time.sleep(RobotConstants.COMMAND_TIME)
            response = self._serial.readline().decode().strip()
            logger.debug(f"Serial RX: {response}")
            return response
        else:
            time.sleep(RobotConstants.COMMAND_TIME)
            return command

    def translate_x(
        self, distance: float, speed: float = RobotConstants.XY_FEED_RATE
    ) -> str:
        g_code = f"G0 X{distance:.2f} F{speed:.2f}"
        response = self.send_gcode(g_code)
        logger.debug(f"translate_x response: {response}")
        return response

    def translate_y(
        self, distance: float, speed: float = RobotConstants.XY_FEED_RATE
    ) -> str:
        g_code = f"G0 Y{distance:.2f} F{speed:.2f}"
        response = self.send_gcode(g_code)
        logger.debug(f"translate_y response: {response}")
        return response

    def translate_z(
        self, distance: float, speed: float = RobotConstants.XY_FEED_RATE
    ) -> str:
        g_code = f"G0 Z{distance:.2f} F{speed:.2f}"
        response = self.send_gcode(g_code)
        logger.debug(f"translate_z response: {response}")
        return response

    def go_to(
        self,
        x_position,
        y_position,
        z_position,
    ) -> str:
        g_code = f"G00 X{x_position:.2f} Y{y_position:.2f} Z{z_position:.2f}\n"
        response = self.send_gcode(g_code)
        return response

    def absolute_mode(self) -> str:
        logger.info("Running in absolute mode.")
        g_code = "G90"
        response = self.send_gcode(g_code)
        return response

    def relative_mode(self) -> str:
        logger.info("Running in relative mode.")
        g_code = "G91"
        response = self.send_gcode(g_code)
        return response

    def get_absolute_position(self) -> str:
        mode_switch = self.absolute_mode()
        logger.debug(f"Mode switch response: {mode_switch}")
        response = None
        if mode_switch.lower() == "ok":
            g_code = "M114"
            response = self.send_gcode(g_code)
            logger.info(f"Absolute position: {response}")
        self.relative_mode()
        return response

    def begin(self) -> str:
        if self._serial:
            self._serial.open()
            response_1 = self.send_gcode(RobotConstants.UNITS)
            if response_1.lower() == "ok":
                return "ok"

            raise IOError("Error during robot initialization; unrecognized command.")

        else:
            logger.debug("DEBUG MODE: start")

    def out_of_way(self) -> None:
        self.go_to(150, 150, z_position=10.0)

    def end(self) -> None:
        if self._serial:
            self.out_of_way()
            self._serial.close()
        else:
            logger.debug("DEBUG MODE: end")
