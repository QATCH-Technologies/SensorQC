import threading
import time
import cv2
import signal
import logging
from DNX64 import DNX64
from constants import DinoLiteConstants

# Configure logging
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def threaded(func):
    """Decorator to run a function in a separate thread."""
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
    return wrapper


class Microscope:
    def __init__(self, microscope_path=DinoLiteConstants.DNX64_PATH, device_index=DinoLiteConstants.DEVICE_INDEX):
        logger.debug("Initializing Microscope.")
        try:
            self._microscope = DNX64(microscope_path)
            logger.info("DNX64 library loaded successfully.")
        except ImportError as err:
            logger.error(f"Failed to load DNX64 library: {err}")
            raise RuntimeError(f"Failed to load DNX64 library: {err}")

        self.__device_index__ = device_index
        self.set_index(self.__device_index__)
        self._microscope.EnableMicroTouch(True)
        logger.debug("MicroTouch enabled.")
        time.sleep(DinoLiteConstants.COMMAND_TIME)
        self._microscope.SetEventCallback(self.microtouch)
        logger.debug("Event callback set.")
        time.sleep(DinoLiteConstants.COMMAND_TIME)
        self.led_off()
        signal.signal(signal.SIGINT, self._handle_exit)

    def _handle_exit(self, signal, frame):
        logger.info("Exiting Microscope application.")
        self.led_off()
        exit(0)

    def enable_microtouch(self):
        logger.debug("Enabling MicroTouch.")
        self._microscope.EnableMicroTouch(True)

    def disable_microtouch(self):
        logger.debug("Disabling MicroTouch.")
        self._microscope.EnableMicroTouch(False)

    def set_index(self, device_index):
        try:
            logger.debug(f"Setting video device index to {device_index}.")
            self._microscope.SetVideoDeviceIndex(device_index)
            time.sleep(DinoLiteConstants.COMMAND_TIME)
        except OSError as e:
            logger.error(
                f"Error retrieving device video index to {device_index}.")
            raise e

    def led_on(self, state):
        logger.debug(f"Turning LED on with state {state}.")
        self.led_off()
        self._microscope.SetLEDState(self.__device_index__, state)
        time.sleep(DinoLiteConstants.COMMAND_TIME)

    def led_off(self):
        logger.debug("Turning LED off.")
        self._microscope.SetLEDState(
            self.__device_index__, DinoLiteConstants.LED_OFF_FLAG
        )
        time.sleep(DinoLiteConstants.COMMAND_TIME)

    def flc_on(self, quadrant=DinoLiteConstants.DEFAULT_FLC_QUADRANT):
        logger.debug(f"Turning FLC on for quadrant {quadrant}.")
        self._microscope.SetFLCSwitch(self.__device_index__, quadrant)

    def flc_off(self):
        logger.debug("Turning FLC off.")
        self._microscope.SetFLCSwitch(
            self.__device_index__, DinoLiteConstants.FLC_OFF
        )

    def flc_level(self, level=DinoLiteConstants.DEFAULT_FLC_LEVEL):
        logger.debug(f"Setting FLC level to {level}.")
        self._microscope.SetFLCLevel(self.__device_index__, level)
        time.sleep(DinoLiteConstants.COMMAND_TIME)

    def microtouch(self, routine):
        logger.debug("MicroTouch event triggered.")
        routine()

    def get_config(self):
        logger.debug("Fetching microscope configuration.")
        config = self._microscope.GetConfig(self.__device_index__)
        config_dict = {
            "config_value": f"0x{config:X}",
            "EDOF": (config & 0x80) == 0x80,
            "AMR": (config & 0x40) == 0x40,
            "eFLC": (config & 0x20) == 0x20,
            "Aim Point Laser": (config & 0x10) == 0x10,
            "2 segments LED": (config & 0xC) == 0x4,
            "3 segments LED": (config & 0xC) == 0x8,
            "FLC": (config & 0x2) == 0x2,
            "AXI": (config & 0x1) == 0x1,
        }
        logger.info(f"Configuration retrieved: {config_dict}")
        time.sleep(DinoLiteConstants.COMMAND_TIME)
        return config_dict


class Camera:
    def __init__(self, debug=False):
        logger.debug("Initializing Camera class.")
        self._camera = cv2.VideoCapture(
            DinoLiteConstants.CAM_INDEX if not debug else 0
        )
        self._camera.set(cv2.CAP_PROP_FPS, DinoLiteConstants.CAMERA_FPS)
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH,
                         DinoLiteConstants.CAMERA_WIDTH)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT,
                         DinoLiteConstants.CAMERA_HEIGHT)
        self.running = False
        logger.info("Camera initialized.")

    def capture_image(self, name="image"):
        logger.debug(f"Capturing image with name prefix {name}.")
        status, frame = self._camera.read()
        if status:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            logger.info(f"Image captured and saved as {filename}.")
        else:
            logger.warning("Failed to capture image.")
        return frame if status else None

    def process_frame(self, frame):
        logger.debug("Processing frame for display.")
        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2
        cv2.line(frame, (center_x - 20, center_y),
                 (center_x + 20, center_y), (0, 0, 255), 2)
        cv2.line(frame, (center_x, center_y - 20),
                 (center_x, center_y + 20), (0, 0, 255), 2)
        return cv2.resize(frame, (DinoLiteConstants.WINDOW_WIDTH, DinoLiteConstants.WINDOW_HEIGHT))

    def run(self):
        if not self._camera.isOpened():
            logger.error("Error opening the camera device.")
            return

        logger.info("Camera streaming started.")
        while self.running:
            status, frame = self._camera.read()
            if status:
                resized_frame = self.process_frame(frame)
                cv2.imshow("Dino-Lite Camera", resized_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Stopping camera stream.")
                    break
        self.release()

    def toggle_camera(self):
        if not self.running:
            logger.info("Starting camera.")
            self.running = True
            threading.Thread(target=self.run, daemon=True).start()
        else:
            logger.info("Stopping camera.")
            self.running = False

    def release(self):
        logger.info("Releasing camera resources.")
        self._camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logger.info("Starting application.")
    microscope = Microscope()
    camera = Camera(debug=True)
    camera.toggle_camera()
