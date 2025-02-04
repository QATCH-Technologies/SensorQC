import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QProgressBar
from PyQt5.QtCore import QThread, pyqtSignal
from dino_lite_edge import Camera, Microscope
from robot import Robot
from constants import CameraConstants, MicroscopeConstants
import numpy as np
import cv2
import time

INLET_POSITION = (0, 0, 0)
Z_RANGE = (0, 1)
Z_STEP = 0.01


class ScanThread(QThread):
    progress_updated = pyqtSignal(int)
    scan_complete = pyqtSignal()

    def __init__(self, rob: Robot, cam: Camera, z_range, z_step):
        super().__init__()
        self.rob = rob
        self.cam = cam
        self.z_range = z_range
        self.z_step = z_step

    def run(self):
        total_steps = len(
            np.arange(self.z_range[0], self.z_range[1], self.z_step))
        current_step = 0

        for z in np.arange(self.z_range[0], self.z_range[1], self.z_step):
            self.rob.go_to(0, 0, z)
            status, image = self.cam._camera.read()
            if status:
                self.calculate_sharpness(image)
            current_step += 1
            progress = int((current_step / total_steps) * 100)
            self.progress_updated.emit(progress)
            time.sleep(0.25)

        self.scan_complete.emit()

    def calculate_sharpness(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        variance = laplacian.var()
        return variance


class ScanUI(QWidget):
    def __init__(self):
        super().__init__()

        self.scan_name = "Default_Scan"
        self.scan_in_progress = False

        self.cam = Camera(debug=True)
        self.scope = Microscope(debug=True)
        self.rob = Robot(debug=True)

        self.scope.led_on(MicroscopeConstants.DARK_FIELD)
        self.rob.begin()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Scan Control Panel')
        self.setGeometry(100, 100, 400, 250)

        self.name_label = QLabel('Scan Name:')
        self.name_input = QLineEdit(self)
        self.name_input.setText(self.scan_name)

        self.start_button = QPushButton('Start Scan', self)
        self.stop_button = QPushButton('Stop Scan', self)
        self.reset_button = QPushButton('Reset Scan', self)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.start_button.clicked.connect(self.start_scan)
        self.stop_button.clicked.connect(self.stop_scan)
        self.reset_button.clicked.connect(self.reset_scan)

        layout = QVBoxLayout()
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_input)
        layout.addWidget(self.progress_bar)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.reset_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def start_scan(self):
        if not self.scan_in_progress:
            self.scan_in_progress = True
            self.scan_name = self.name_input.text()
            self.progress_bar.setValue(0)
            self.scan_thread = ScanThread(self.rob, self.cam, Z_RANGE, Z_STEP)
            self.scan_thread.progress_updated.connect(self.update_progress)
            self.scan_thread.scan_complete.connect(self.scan_complete)
            self.scan_thread.start()
            print(f"Scan started: {self.scan_name}")
        else:
            print("Scan is already in progress.")

    def stop_scan(self):
        if self.scan_in_progress:
            self.rob.out_of_way()
            self.scan_thread.terminate()
            self.scan_in_progress = False
            self.progress_bar.setValue(0)
            print(f"Scan stopped: {self.scan_name}")
        else:
            print("No scan in progress.")

    def reset_scan(self):
        if self.scan_in_progress:
            self.scan_thread.terminate()
            self.scan_in_progress = False
        self.rob.go_to(INLET_POSITION[0], INLET_POSITION[1], INLET_POSITION[2])
        time.sleep(1)
        self.name_input.setText("Sensor_")
        self.progress_bar.setValue(0)
        print("Scan reset.")

    def update_progress(self, progress):
        self.progress_bar.setValue(progress)

    def scan_complete(self):
        self.scan_in_progress = False
        print("Scan completed.")

    def get_inlet(self, robot: Robot, camera: Camera, z_range, z_step):
        best_sharpness = 0
        best_z_position = 0
        best_image = None

        for z in np.arange(z_range[0], z_range[1], z_step):
            robot.go_to(INLET_POSITION[0], INLET_POSITION[1], z)
            status, image = camera._camera.read()
            if status:
                sharpness = self.calculate_sharpness(image)
                if sharpness > best_sharpness:
                    best_sharpness = sharpness
                    best_z_position = z
                    best_image = image
            time.sleep(0.25)
        return best_image

    def calculate_sharpness(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        variance = laplacian.var()
        return variance

    def closeEvent(self, event):
        self.rob.out_of_way()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = ScanUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
