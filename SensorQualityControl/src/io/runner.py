from scanner import TileScanner
from calibrate import AutofocusCalibrator
from constants import SystemConstants

if __name__ == "__main__":
    calibrator = AutofocusCalibrator()
    calibrator.run_calibration()
    scanner = TileScanner()
    scanner.run(z_points=SystemConstants.FOCUS_PLANE_POINTS)
