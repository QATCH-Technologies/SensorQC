from enum import Enum
from collections import namedtuple


class SystemConstants:
    """System-wide constants related to positions, focus, and ranges."""

    # Named tuple for positions
    Position = namedtuple("Position", ["x", "y", "z", "rotation"])
    FocusPoint = namedtuple("FocusPoint", ["x", "y", "z"])

    INITIAL_POSITION = Position(108.2, 130.9, 6.19, 0.00)
    FINAL_POSITION = Position(119.6, 119.4, 4.75, 0.00)
    TOP_LEFT_FOCUS = FocusPoint(109.5, 129.5, 4.90)
    TOP_RIGHT_FOCUS = FocusPoint(117.1, 128.3, 5.00)
    BOTTOM_LEFT_FOCUS = FocusPoint(117.1, 121.1, 4.90)
    BOTTOM_RIGHT_FOCUS = FocusPoint(109.5, 122.5, 4.95)

    FOCUS_PLANE_POINTS = [
        TOP_LEFT_FOCUS,  # Top-left
        TOP_RIGHT_FOCUS,  # Top-right
        BOTTOM_LEFT_FOCUS,  # Bottom-left
        BOTTOM_RIGHT_FOCUS,  # Bottom-right
    ]

    FOCUS_RANGE = (3.0, 6.0)  # Min and max focus range
    FOCUS_STEP = 0.05  # Incremental focus step size

    X_DELTA = 0.5
    Y_DELTA = -0.5

    @classmethod
    def validate_focus_range(cls):
        """Ensures focus range is valid."""
        min_focus, max_focus = cls.FOCUS_RANGE
        if min_focus >= max_focus:
            raise ValueError(
                "FOCUS_RANGE must have a minimum less than the maximum.")


class RobotConstants:
    """Constants related to robot configuration and operation."""

    class Units(Enum):
        METRIC = "G21"
        IMPERIAL = "G20"

    HOMING_TIME = 17  # Time in seconds for homing
    UNITS = Units.METRIC.value  # Default units
    FEED_RATE = 1000  # Movement speed in mm/min
    COMMAND_TIME = 0.5  # Time between commands in seconds
    BAUDRATE = 115200  # Communication speed


class CameraConstants:
    """Constants for camera configuration."""

    class AutoExposure(Enum):
        ON = 1
        OFF = 0

    CAMERA_INDEX = 1  # Default camera index
    AUTOEXPOSURE_VALUE = 414  # Autoexposure brightness value
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
    QUERY_TIME = 0.05  # Time for queries in seconds
    COMMAND_TIME = 0.25  # Time between commands in seconds
