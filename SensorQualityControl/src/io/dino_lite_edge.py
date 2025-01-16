import importlib
import math
import threading
import time
import cv2
from DNX64 import *
import signal
import atexit
from constants import CameraConstants, MicroscopeConstants


def threaded(func):
    """Wrapper to run a function in a separate thread with @threaded decorator"""

    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()

    return wrapper


class Microscope:
    def __init__(self, microscope_path: str = MicroscopeConstants.DNX64_PATH, device_index: int = MicroscopeConstants.DEVICE_INDEX):
        try:
            DNX64 = getattr(importlib.import_module("DNX64"), "DNX64")
        except ImportError as err:
            print("Error: ", err)
            raise

        self._device_index = device_index
        self._microscope = DNX64(microscope_path)

        # Initialize microscope and set default states
        self.set_index(self._device_index)
        self._microscope.EnableMicroTouch(True)
        time.sleep(MicroscopeConstants.COMMAND_TIME)
        self._microscope.SetEventCallback(self.microtouch)
        time.sleep(MicroscopeConstants.COMMAND_TIME)

        # Ensure the LED is off initially
        self.led_off()

        # Register cleanup handlers
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)
        atexit.register(self._cleanup)

    def _handle_exit(self, signum, frame):
        """Handle termination signals to ensure cleanup."""
        print(f"Signal {signum} received. Turning off LED and exiting.")
        self._cleanup()
        exit(0)

    def _cleanup(self):
        """Cleanup resources and ensure LED is off."""
        self.led_off()
        print("LED turned off. Resources cleaned up.")

    def enable_microtouch(self):
        return self._microscope.EnableMicroTouch(True)

    def disable_microtouch(self):
        return self._microscope.EnableMicroTouch(False)

    def auto_exposure(self, state: int):
        self._microscope.SetAutoExposure(self._device_index, state)

    def flc_on(self, quadrant: int = MicroscopeConstants.DEFAULT_FLC_QUADRANT):
        self._microscope.SetFLCSwitch(self._device_index, quadrant)

    def flc_off(self):
        self._microscope.SetFLCSwitch(
            self._device_index, MicroscopeConstants.FLC_OFF)

    def flc_level(self, level: int = MicroscopeConstants.DEFAULT_FLC_LEVEL):
        self._microscope.SetFLCLevel(self._device_index, level)
        time.sleep(MicroscopeConstants.COMMAND_TIME)

    @threaded
    def led_on(self, state):
        self.led_off()
        self._microscope.SetLEDState(self._device_index, state)
        time.sleep(MicroscopeConstants.COMMAND_TIME)

    def microtouch(self, routine):
        routine()

    def led_off(self):
        self._microscope.SetLEDState(
            self._device_index, MicroscopeConstants.LED_OFF)
        time.sleep(MicroscopeConstants.COMMAND_TIME)

    def set_index(self, device_index: int = MicroscopeConstants.DEVICE_INDEX):
        self._microscope.SetVideoDeviceIndex(device_index)
        time.sleep(MicroscopeConstants.COMMAND_TIME)

    def set_exposure(self, exposure: int):
        self._microscope.SetExposureValue(self._device_index, exposure)

    def set_autoexposure(self, state: int):
        self._microscope.SetAutoExposure(self._device_index, state)

    def set_autoexposure_target(self, target: int):
        self._microscope.SetAETarget(self._device_index, target)

    def get_id(self):
        id = self._microscope.GetDeviceId(self._device_index)
        time.sleep(MicroscopeConstants.QUERY_TIME)
        return id

    def get_exposure(self):
        return self._microscope.GetExposureValue(self._device_index)

    def get_autoexposure(self):
        return self._microscope.GetAutoExposure(self._device_index)

    def get_autoexposure_target(self):
        return self._microscope.GetAETarget(self._device_index)

    def get_ida(self):
        return self._microscope.GetDeviceIDA(self._device_index)

    def get_config(self):
        config = self._microscope.GetConfig(self._device_index)
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
        time.sleep(MicroscopeConstants.QUERY_TIME)
        return config_dict

    def get_fov_index(self):
        amr = self._microscope.GetAMR(MicroscopeConstants.DEVICE_INDEX)
        fov = self._microscope.FOVx(MicroscopeConstants.DEVICE_INDEX, amr)
        amr = round(amr, 1)
        fov = round(fov / 1000, 2)

        if fov == math.inf:
            fov = round(self._microscope.FOVx(
                MicroscopeConstants.DEVICE_INDEX, 50.0) / 1000.0, 2)
            fov_info = {"magnification": 50.0, "fov_um": fov}
        else:
            fov_info = {"magnification": amr, "fov_um": fov}

        time.sleep(MicroscopeConstants.QUERY_TIME)
        return fov_info

    def get_amr(self):
        config = self._microscope.GetConfig(self._device_index)
        amr_info = {}

        if (config & 0x40) == 0x40:
            amr = self._microscope.GetAMR(self._device_index)
            amr = round(amr, 1)
            amr_info = {"amr_value": amr, "message": f"{amr}x"}
        else:
            amr_info = {
                "amr_value": None,
                "message": "It does not belong to the AMR series.",
            }

        time.sleep(MicroscopeConstants.QUERY_TIME)
        return amr_info

    def end(self):
        self.led_off()


class Camera:
    def __init__(self, recording: bool = False, video_writer=None, debug=False):
        if debug:
            self._camera = cv2.VideoCapture(0)
        else:
            self._recording = recording
            self._video_writer = video_writer

            self._camera = cv2.VideoCapture(CameraConstants.CAMERA_INDEX)
            self._camera.set(cv2.CAP_PROP_FPS, CameraConstants.CAMERA_FPS)
            self._camera.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("m", "j", "p", "g")
            )
            self._camera.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G")
            )
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH,
                             CameraConstants.CAMERA_RESOLUTIONS.get("1280x960")[0])
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT,
                             CameraConstants.CAMERA_RESOLUTIONS.get("1280x960")[1])
            self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = False

    def set_index(self, microscope):
        microscope.SetVideoDeviceIndex(0)
        time.sleep(MicroscopeConstants.COMMAND_TIME)

    def capture_image(self, name: str):
        """Capture an image and save it in the current working directory."""
        status, frame = self._camera.read()

        if status:
            self.running = True
            if name == "":
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"image_{timestamp}.jpg"
            else:
                filename = f"{name}.jpg"
            cv2.imwrite(filename, frame)
        self.running = False
        return frame

    def process_frame_w_crosshairs(self, frame):
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
        return cv2.resize(frame, (CameraConstants.CAMERA_RESOLUTIONS.get("1280x960")[0], CameraConstants.CAMERA_RESOLUTIONS.get("1280x960")[1]))

    def run(self):
        if not self._camera.isOpened():
            print("Error opening the camera device.")
            return

        while True:
            status, frame = self._camera.read()
            if status:
                resized_frame = self.process_frame_w_crosshairs(frame)
                cv2.imshow("Dino-Lite Camera", resized_frame)
                if cv2.waitKey(1) & 0xFF == ord("p"):
                    break

    def toggle_camera(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.run, daemon=True).start()
        else:
            self.running = False
            self._camera.release()
            cv2.destroyAllWindows()

    def release(self):
        if self._camera.isOpened():
            self._camera.release()


if __name__ == "__main__":
    cam = Camera(debug=True)
