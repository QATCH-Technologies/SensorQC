from enum import Enum
from collections import namedtuple
from positions import Position


class SystemConstants:
    """System-wide constants related to positions, focus, and ranges."""

    # Named tuple for positions
    # Position = namedtuple("Position", ["x", "y", "z", "rotation"])
    # FocusPoint = namedtuple("FocusPoint", ["x", "y"])

    INITIAL_POSITION = Position(
        x=108.2, y=130.9, z=6.19, location_name="Initial_Position")
    FINAL_POSITION = Position(x=119.6, y=119.4, z=4.75,
                              location_name="Final_Position")
    TOP_LEFT_FOCUS = Position(y=0, x=0, z=0,
                              location_name="Top_Left_Focus_Point")
    TOP_RIGHT_FOCUS = Position(
        x=117.1, y=128.3, z=0, location_name="Top_Right_Focus_Point")
    BOTTOM_LEFT_FOCUS = Position(
        x=117.1, y=121.1, z=0, location_name="Bottom_Left_Focus_Point")
    BOTTOM_RIGHT_FOCUS = Position(
        x=109.5, y=122.5, z=0, location_name="Bottom_Right_Focus_Point")
    LABEL_FOCUS = Position(x=109.5, y=129.5, z=0,
                           location_name="Label_Focus_Point")
    INLET_FOCUS = Position(y=0, x=0, z=0,
                           location_name="Inlet_Focus_Point")
    OUTLET_FOCUS = Position(y=0, x=0, z=0,
                            location_name="Outlet_Focus_Point")
    CHANNEL_1_FOCUS = Position(y=0, x=0, z=0,
                               location_name="Channel_1_Focus_Point")
    CHANNEL_2_FOCUS = Position(y=0, x=0, z=0,
                               location_name="Channel_2_Focus_Point")
    FOCUS_PLANE_POINTS = [
        LABEL_FOCUS,
        TOP_LEFT_FOCUS,
        TOP_RIGHT_FOCUS,
        BOTTOM_LEFT_FOCUS,
        BOTTOM_RIGHT_FOCUS,
        INLET_FOCUS,
        OUTLET_FOCUS,
        CHANNEL_1_FOCUS,
        CHANNEL_2_FOCUS,
    ]

    FOCUS_RANGE = (3.0, 6.0)  # Min and max focus range
    FOCUS_STEP = 0.05  # Incremental focus step size

    X_DELTA = 0.5
    Y_DELTA = -0.5
    DEBUG = True

    @classmethod
    def validate_focus_range(cls):
        """Ensures focus range is valid."""
        min_focus, max_focus = cls.FOCUS_RANGE
        if min_focus >= max_focus:
            raise ValueError(
                "FOCUS_RANGE must have a minimum less than the maximum.")


class RobotConstants:
    """Constants related to robot configuration and operation."""
    ROBOT_PORT = "COM4"

    class Units(Enum):
        METRIC = "G21"
        IMPERIAL = "G20"

    HOMING_TIME = 17  # Time in seconds for homing
    UNITS = Units.METRIC.value  # Default units
    FEED_RATE = 1000  # Movement speed in mm/min
    COMMAND_TIME = 0.5  # Time between commands in seconds
    BAUDRATE = 115200  # Communication speed
    COLUMN_DELAY = 0.1
    ROW_DELAY = 2


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
