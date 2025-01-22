from scanner import TileScanner
from calibrate import AutofocusCalibrator
from constants import SystemConstants
from robot import Robot
from dino_lite_edge import Microscope, Camera
from constants import RobotConstants, MicroscopeConstants
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ScanRunner:
    @staticmethod
    def run():
        scope = Microscope(debug=SystemConstants.DEBUG)
        cam = Camera(debug=SystemConstants.DEBUG)
        rob = Robot(port=RobotConstants.ROBOT_PORT,
                    debug=SystemConstants.DEBUG)
        rob.begin()
        rob.absolute_mode()
        rob.home()
        scope.led_on(state=MicroscopeConstants.DARK_FIELD)
        calibrator = AutofocusCalibrator(
            camera=cam, microscope=scope, robot=rob)
        calibrator.run_calibration()
        scanner = TileScanner(camera=cam, microscope=scope, robot=rob)
        scanner.run(z_points=SystemConstants.FOCUS_PLANE_POINTS)
        cam.release()
        scope.end()
        rob.end()


if __name__ == "__main__":
    ScanRunner.run()
