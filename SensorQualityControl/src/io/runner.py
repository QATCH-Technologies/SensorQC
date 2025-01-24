from scanner import TileScanner
from calibrate import AutofocusCalibrator
from constants import SystemConstants
from robot import Robot
from dino_lite_edge import Microscope, Camera
from constants import RobotConstants, CameraConstants
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ScanRunner:
    @staticmethod
    def run():
        scope = Microscope(debug=SystemConstants.DEBUG)
        scope.set_autoexposure(CameraConstants.AUTOEXPOSURE_OFF)
        scope.set_exposure(CameraConstants.DF_AUTOEXPOSURE_VALUE)
        # scope.led_on(state=MicroscopeConstants.DARK_FIELD)
        cam = Camera(debug=SystemConstants.DEBUG)
        rob = Robot(port=RobotConstants.ROBOT_PORT, debug=SystemConstants.DEBUG)
        rob.begin()
        rob.absolute_mode()
        # rob.home()
        while True:
            runname = input(">>> Runname: ")
            # calibrator = AutofocusCalibrator(camera=cam, microscope=scope, robot=rob)
            # calibrator.run_calibration()
            scanner = TileScanner(camera=cam, microscope=scope, robot=rob)
            scanner.run(z_points=SystemConstants.FOCUS_PLANE_POINTS, runname=runname)
            cont = input(
                "Run complete, press any key to continue to next run or 'q' to quit."
            )
            if cont == "q":
                break
        cam.release()
        scope.end()
        rob.end()


if __name__ == "__main__":
    ScanRunner.run()
