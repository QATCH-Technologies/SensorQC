import importlib
import math
import threading
import time
import cv2
from DNX64 import *
import numpy as np

DNX64_PATH = "C:\\Program Files\\DNX64\\DNX64.dll"
# Global variables
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 960
"""
Supported Resolutions:
- 640 x 480
- 1280 x 960
- 1600 x 1200
- 2048 x 1536
- 2582 x 1944
"""
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 1280, 960, 30
DEVICE_INDEX = 0
# Camera index, please change it if you have more than one camera,
# i.e. webcam, connected to your PC until CAM_INDEX is been set to first Dino-Lite product.
CAM_INDEX = 1
# Buffer time for Dino-Lite to return value
QUERY_TIME = 0.05
# Buffer time to allow Dino-Lite to process command
COMMAND_TIME = 0.25

LED_OFF_FLAG = 0
BRIGHT_FIELD_FLAG = 1
DARK_FIELD_FLAG = 2
DEFAULT_FLC_LEVEL = 3
DEFAULT_FLC_QUADRANT = 15
FLC_OFF = 16


def threaded(func):
    """Wrapper to run a function in a separate thread with @threaded decorator"""

    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()

    return wrapper


class Microscope:
    def __init__(
        self, microscope_path: str = DNX64_PATH, device_index: int = DEVICE_INDEX
    ):
        try:
            DNX64 = getattr(importlib.import_module("DNX64"), "DNX64")
        except ImportError as err:
            print("Error: ", err)
        # Set index of video device. Call before Init().
        self.__device_index__ = device_index
        self.__microscope__ = DNX64(microscope_path)
        self.set_index(self.__device_index__)
        # Enabled MicroTouch Event
        self.__microscope__.EnableMicroTouch(True)
        time.sleep(COMMAND_TIME)
        # Function to execute when MicroTouch event detected
        self.__microscope__.SetEventCallback(self.microtouch)
        time.sleep(COMMAND_TIME)
        self.led_off()

    def enable_microtouch(self):
        return self.__microscope__.EnableMicroTouch(True)

    def disable_microtouch(self):
        return self.__microscope__.EnableMicroTouch(False)

    def auto_exposure(self, state: int):
        self.__microscope__.SetAutoExposure(self.__device_index__, state)

    def flc_on(self, quadrant: int = DEFAULT_FLC_QUADRANT):
        self.__microscope__.SetFLCSwitch(self.__device_index__, quadrant)

    def flc_off(self):
        self.__microscope__.SetFLCSwitch(self.__device_index__, FLC_OFF)

    def flc_level(self, level: int = DEFAULT_FLC_LEVEL):
        self.__microscope__.SetFLCLevel(self.__device_index__, level)
        time.sleep(COMMAND_TIME)

    @threaded
    def led_on(self, state):
        self.led_off()
        self.__microscope__.SetLEDState(self.__device_index__, state)
        time.sleep(COMMAND_TIME)

    def microtouch(self, routine):
        routine()

    def led_off(self):
        self.__microscope__.SetLEDState(self.__device_index__, LED_OFF_FLAG)
        time.sleep(COMMAND_TIME)

    def set_index(self, device_index: int = DEVICE_INDEX):
        self.__microscope__.SetVideoDeviceIndex(device_index)
        time.sleep(COMMAND_TIME)

    def set_exposure(self, exposure: int):
        self.__microscope__.SetExposureValue(self.__device_index__, exposure)

    def set_autoexposure(self, state: int):
        self.__microscope__.SetAutoExposure(self.__device_index__, state)

    def set_autoexposure_target(self, target: int):
        self.__microscope__.SetAETarget(self.__device_index__, target)

    def get_id(self):
        id = self.__microscope__.GetDeviceId(self.__device_index__)
        time.sleep(QUERY_TIME)
        return id

    def get_exposure(self):
        return self.__microscope__.GetExposureValue(self.__device_index__)

    def get_autoexposure(self):
        return self.__microscope__.GetAutoExposure(self.__device_index__)

    def get_autoexposure_target(self):
        return self.__microscope__.GetAETarget(self.__device_index__)

    def get_ida(self):
        return self.__microscope__.GetDeviceIDA(self.__device_index__)

    def get_config(self):
        config = self.__microscope__.GetConfig(self.__device_index__)
        config_dict = {
            "config_value": "0x{:X}".format(config),
            "EDOF": (config & 0x80) == 0x80,
            "AMR": (config & 0x40) == 0x40,
            "eFLC": (config & 0x20) == 0x20,
            "Aim Point Laser": (config & 0x10) == 0x10,
            "2 segments LED": (config & 0xC) == 0x4,
            "3 segments LED": (config & 0xC) == 0x8,
            "FLC": (config & 0x2) == 0x2,
            "AXI": (config & 0x1) == 0x1,
        }
        time.sleep(QUERY_TIME)
        return config_dict

    def get_fov_index(self):
        amr = self.__microscope__.GetAMR(DEVICE_INDEX)
        fov = self.__microscope__.FOVx(DEVICE_INDEX, amr)
        amr = round(amr, 1)
        fov = round(fov / 1000, 2)

        if fov == math.inf:
            fov = round(self.__microscope__.FOVx(DEVICE_INDEX, 50.0) / 1000.0, 2)
            fov_info = {"magnification": 50.0, "fov_um": fov}
        else:
            fov_info = {"magnification": amr, "fov_um": fov}

        time.sleep(QUERY_TIME)
        return fov_info

    def get_amr(self):
        config = self.__microscope__.GetConfig(self.__device_index__)
        amr_info = {}

        if (config & 0x40) == 0x40:
            amr = self.__microscope__.GetAMR(self.__device_index__)
            amr = round(amr, 1)
            amr_info = {"amr_value": amr, "message": f"{amr}x"}
        else:
            amr_info = {
                "amr_value": None,
                "message": "It does not belong to the AMR series.",
            }

        time.sleep(QUERY_TIME)
        return amr_info

    def end(self):
        self.led_off()


class Camera:
    def __init__(self, recording: bool = False, video_writer=None, debug=False):
        if debug:
            self.__camera__ = cv2.VideoCapture(0)
        else:
            self.__recording__ = recording
            self.__video_writer__ = video_writer

            self.__camera__ = cv2.VideoCapture(CAM_INDEX)
            self.__camera__.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            self.__camera__.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("m", "j", "p", "g")
            )
            self.__camera__.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G")
            )
            self.__camera__.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.__camera__.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.__camera__.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = False

    def set_index(microscope):
        microscope.SetVideoDeviceIndex(0)
        time.sleep(COMMAND_TIME)

    def straighten_image(self, image):
        return image

    def capture_image(self, name: str = ""):
        """Capture an image and save it in the current working directory."""
        status, frame = self.__camera__.read()

        if status:
            self.running = True
            if name == "":
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"image_{timestamp}.jpg"
            else:
                filename = f"{name}.jpg"
            self.straighten_image(frame)
            cv2.imwrite(filename, frame)
        self.running = False

    def process_frame(self, frame):
        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2

        cv2.line(
            frame,
            (center_x - 20, center_y),
            (center_x + 20, center_y),
            (0, 0, 255),
            2,
        )
        cv2.line(
            frame,
            (center_x, center_y - 20),
            (center_x, center_y + 20),
            (0, 0, 255),
            2,
        )
        return cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

    def run(self):
        if not self.__camera__.isOpened():
            print("Error opening the camera device.")
            return

        while True:
            status, frame = self.__camera__.read()
            if status:
                resized_frame = self.process_frame(frame)
                cv2.imshow("Dino-Lite Camera", resized_frame)
                if cv2.waitKey(1) & 0xFF == ord("p"):
                    break

    def toggle_camera(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.run, daemon=True).start()
        else:
            self.running = False
            self.__camera__.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    cam = Camera(debug=True)
