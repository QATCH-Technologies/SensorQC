from dino_lite_edge import Microscope, Camera
from constants import MicroscopeConstants, CameraConstants

if __name__ == "__main__":
    while True:
        scope = Microscope()
        scope.led_on(MicroscopeConstants.DARK_FIELD)
        scope.auto_exposure(CameraConstants.AUTOEXPOSURE_ON)
        print(scope.get_exposure(), scope.get_autoexposure())
