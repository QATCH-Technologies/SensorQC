from enum import Enum
import numpy as np


class SystemConstants:
    X_DELTA = (126.10 - 119.20) / 2
    Y_DELTA = 136.00 - 132.20
    TILE_DIMENSIONS = (X_DELTA, Y_DELTA)
    DEBUG = False

    SERVER_PATH = r"C:\Users\QATCH\Documents\SVN Repos\SensorQC"

    Z_HEIGHT = 30.00

    TOP_LEFT = (122.7, 134.0)
    BOTTOM_RIGHT = (134.0 + X_DELTA, 123.0 - Y_DELTA)
    NUM_VIDEO_CAPTURE_DEVICES = 2
    TILE_TO_TILE_DELAY = 1
    SENSOR_DIMESNIONS = (10.85, 11.40)
    z_values = np.array([6.5, 30.0])
    width_values = np.array([1.4, 6.9])
    height_values = np.array([1.0, 3.8])
    LABEL_POS = (122.8, 133.5)


@classmethod
def validate_focus_range(cls):
    """Ensures focus range is valid."""
    min_focus, max_focus = cls.FOCUS_RANGE
    if min_focus >= max_focus:
        raise ValueError("FOCUS_RANGE must have a minimum less than the maximum.")


class RobotConstants:
    """Constants related to robot configuration and operation."""

    ROBOT_PORT = "COM4"

    class Units(Enum):
        METRIC = "G21"
        IMPERIAL = "G20"

    HOMING_TIME = 17  # Time in seconds for homing
    UNITS = Units.METRIC.value  # Default units
    XY_FEED_RATE = 600  # Movement speed in mm/min
    Z_FEED_RATE = 20
    COMMAND_TIME = 0.025  # Time between commands in seconds
    BAUDRATE = 115200  # Communication speed
    COLUMN_DELAY = 0.4
    ROW_DELAY = 2
    Y_MAX = 220.0
    X_MAX = 220.0


class CameraConstants:
    """Constants for camera configuration."""

    class AutoExposure(Enum):
        ON = 1
        OFF = 0

    CAMERA_INDEX = 1  # Default camera index
    BF_AUTOEXPOSURE_VALUE = 414
    DF_AUTOEXPOSURE_VALUE = 90
    AUTOEXPOSURE_ON = AutoExposure.ON.value
    AUTOEXPOSURE_OFF = AutoExposure.OFF.value
    CAMERA_FPS = 30  # Frames per second

    CAMERA_RESOLUTIONS = {
        "640x480": (640, 480),
        "1280x960": (1280, 960),
        "1600x1200": (1600, 1200),
        "2048x1536": (2048, 1536),
        "2582x1944": (2582, 1944),
    }


class MicroscopeConstants:
    """Constants for microscope configuration."""

    class LEDMode(Enum):
        OFF = 0
        BRIGHT_FIELD = 1
        DARK_FIELD = 2

    DNX64_PATH = "C:\\Program Files\\DNX64\\DNX64.dll"
    DEVICE_INDEX = 0  # Default device index

    LED_OFF = LEDMode.OFF.value
    BRIGHT_FIELD = LEDMode.BRIGHT_FIELD.value
    DARK_FIELD = LEDMode.DARK_FIELD.value

    DEFAULT_FLC_LEVEL = 3
    DEFAULT_FLC_QUADRANT = 15
    FLC_OFF = 16
    QUERY_TIME = 0.05
    COMMAND_TIME = 0.05
    NAME = "Dino-Lite Edge"
