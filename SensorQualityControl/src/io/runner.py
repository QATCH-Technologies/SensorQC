from scanner import TileScanner
from calibrate import AutofocusCalibrator
from constants import SystemConstants
import os
if __name__ == "__main__":
    calibrator = AutofocusCalibrator()
    # calibrator.run_calibration()
    scanner = TileScanner()
    scanner.run(folder_path=os.path.join(
        "content", "images", "df_c"), z_points=SystemConstants.FOCUS_PLANE_POINTS)
