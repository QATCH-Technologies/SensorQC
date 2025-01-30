import cv2
import numpy as np
import json
import logging
from dino_lite_edge import Camera, Microscope
from robot import Robot
from constants import SystemConstants, RobotConstants
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


class AutofocusCalibrator:
    def __init__(self, microscope: Microscope, camera: Camera, robot: Robot):
        self._scope = microscope
        self._camera = camera
        self._robot = robot

    def init_params(self):
        self._robot.go_to(
            SystemConstants.INITIAL_POSITION.x,
            SystemConstants.INITIAL_POSITION.y,
            SystemConstants.INITIAL_POSITION.z,
        )
        logger.info("Gantry has reached the initial position.")
        input("Wait for gantry to stop moving. Press any key to proceed.")

    @staticmethod
    def calculate_laplacian_variance(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        variance = laplacian.var()
        return variance

    def autofocus(self, z_range, step_size):
        z_min, z_max = z_range
        best_z = z_min
        best_frame = None
        max_sharpness = 0

        for z in np.arange(z_min, z_max, step_size):
            logger.debug(f"Moving to Z={z} for autofocus check...")
            self._robot.translate_z(z)
            time.sleep(0.1)
            status, frame = self._camera._camera.read()
            if not status:
                # logger.warning("Failed to capture image.")
                continue

            sharpness = self.calculate_laplacian_variance(frame)
            logger.debug(f"Laplacian variance (sharpness) at Z={z}: {sharpness}")

            if sharpness > max_sharpness:
                max_sharpness = sharpness
                best_z = z
                best_frame = frame

        logger.debug(f"Best Z-height for focus: {best_z}, Sharpness: {max_sharpness}")
        return best_z, best_frame

    def calibrate_focus(self, focus_positions: list, z_range, step_size):
        z_heights = {}
        for position in tqdm(focus_positions, desc="Calibrating focus"):
            time.sleep(RobotConstants.ROW_DELAY)
            logger.debug(
                f"Moving to {position.location_name}: (X={position.x}, Y={position.y})"
            )
            self._robot.go_to(position.x, position.y, z_range[0])
            logger.debug(f"Running autofocus at {position.location_name}...")
            z_height, best_frame = self.autofocus(z_range, step_size)
            cal_filename = f"af_{position.location_name}.jpg"
            cv2.imwrite(cal_filename, best_frame)
            position.z = z_height
            z_heights[position.location_name] = z_height

        return z_heights

    @staticmethod
    def save_results_to_json(results, file_name="focus_config.json"):
        with open(file_name, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {file_name}")

    def run_calibration(self):
        try:
            self.init_params()
            z_height_results = self.calibrate_focus(
                SystemConstants.FOCUS_PLANE_POINTS,
                SystemConstants.FOCUS_RANGE,
                SystemConstants.FOCUS_STEP,
            )
            logger.info("Calibration Results (Z-heights):")
            logger.info(z_height_results)
            self.save_results_to_json(z_height_results)
        except KeyboardInterrupt:
            print("Process interrupted by user.")


if __name__ == "__main__":
    calibrator = AutofocusCalibrator()
    calibrator.run_calibration()
