import sys
import os
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QProgressBar,
    QFileDialog,
)
from PyQt5.QtCore import QThread, pyqtSignal
from dino_lite_edge import Camera, Microscope
from robot import Robot
from constants import MicroscopeConstants
import numpy as np
import cv2
import time
from PyQt5.QtGui import QPixmap, QImage
import logging as log

Z_RANGE = (10.7, 11)
INLET_POSITION = (111.9, 125.2, Z_RANGE[0])
Z_STEP = 0.05


class ScanThread(QThread):
    progress_updated = pyqtSignal(int)
    scan_complete = pyqtSignal(np.ndarray)

    def __init__(self, rob: Robot, cam: Camera, z_range, z_step, scan_name, save_path):
        super().__init__()
        self.rob = rob
        self.cam = cam
        self.z_range = z_range
        self.z_step = z_step
        self.scan_name = scan_name
        self.save_path = save_path
        self.best_image = None
        self.best_sharpness = -np.inf

    def run(self):
        total_steps = len(
            np.arange(self.z_range[0], self.z_range[1], self.z_step))
        current_step = 0
        self.rob.go_to(INLET_POSITION[0], INLET_POSITION[1], INLET_POSITION[2])
        time.sleep(4)

        for z in np.arange(self.z_range[0], self.z_range[1], self.z_step):
            self.rob.go_to(INLET_POSITION[0], INLET_POSITION[1], z)
            time.sleep(0.5)
            status, image = self.cam._camera.read()
            if status:
                sharpness = self.calculate_sharpness(image)
                if sharpness > self.best_sharpness:
                    self.best_sharpness = sharpness
                    self.best_image = image.copy()

            current_step += 1
            progress = int((current_step / total_steps) * 100)
            self.progress_updated.emit(progress)

        if self.best_image is not None:
            self.best_image = self.straighten_image()
            file_path = os.path.join(self.save_path, f"{self.scan_name}.jpg")
            cv2.imwrite(file_path, self.best_image)
            print(f"Best image saved at: {file_path}")

        self.scan_complete.emit(self.best_image)

    def apply_scale(self, image):
        height, width, _ = image.shape

        # Grid spacing in pixels
        major_x_spacing = 304  # Major grid: 0.1mm in X
        major_y_spacing = 324  # Major grid: 0.1mm in Y

        minor_x_spacing = major_x_spacing // 2  # Minor grid: 0.05mm in X
        minor_y_spacing = major_y_spacing // 2  # Minor grid: 0.05mm in Y

        major_grid_color = (235, 90, 60)
        minor_grid_color = (223, 151, 85)
        text_color = (235, 90, 60)
        thickness_major = 2
        thickness_minor = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5

        # Draw minor grid lines (lighter)
        for x in range(0, width, minor_x_spacing):
            cv2.line(image, (x, 0), (x, height),
                     minor_grid_color, thickness_minor)

        for y in range(0, height, minor_y_spacing):
            cv2.line(image, (0, y), (width, y),
                     minor_grid_color, thickness_minor)

        # Draw major grid lines (bolder)
        for x in range(0, width, major_x_spacing):
            cv2.line(image, (x, 0), (x, height),
                     major_grid_color, thickness_major)

        for y in range(0, height, major_y_spacing):
            cv2.line(image, (0, y), (width, y),
                     major_grid_color, thickness_major)

        # Add annotations at major grid intersections
        for x in range(0, width, major_x_spacing):
            for y in range(0, height, major_y_spacing):
                label = f"({x//major_x_spacing * 0.1:.1f}, {y//major_y_spacing * 0.1:.1f}) mm"
                cv2.putText(image, label, (x + 5, y + 15),
                            font, font_scale, text_color, 1)

    def straighten_image(self):
        gray = cv2.cvtColor(self.best_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[-1]
            if angle < -45:
                angle += 90

            # Get the rotation matrix
            (h, w) = self.best_image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            straightened = cv2.warpAffine(
                self.best_image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return self.apply_scale(straightened)
        else:
            print("No contours found, skipping straightening.")
            return self.apply_scale(self.best_image)

    def calculate_sharpness(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        variance = laplacian.var()
        return variance


class ScanUI(QWidget):
    def __init__(self):
        super().__init__()
        self.scan_name = "Sensor_"
        self.save_path = os.getcwd()  # Default save path is the current directory
        self.scan_in_progress = False

        self.cam = Camera()
        self.scope = Microscope()
        self.rob = Robot()

        self.scope.led_on(MicroscopeConstants.DARK_FIELD)
        self.rob.begin()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Scan Control Panel")
        self.setGeometry(100, 100, 400, 400)

        self.name_label = QLabel("Scan Name:")
        self.name_input = QLineEdit(self)
        self.name_input.setText(self.scan_name)

        self.path_label = QLabel(f"Save Path: {self.save_path}")
        self.browse_button = QPushButton("Browse", self)
        self.browse_button.clicked.connect(self.browse_folder)

        self.start_button = QPushButton("Start Scan", self)
        self.stop_button = QPushButton("Stop Scan", self)
        self.reset_button = QPushButton("Reset Scan", self)
        self.home_button = QPushButton("Home", self)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.image_label = QLabel("Best Image will appear here")
        self.image_label.setFixedSize(320, 240)

        self.start_button.clicked.connect(self.start_scan)
        self.stop_button.clicked.connect(self.stop_scan)
        self.reset_button.clicked.connect(self.reset_scan)
        self.home_button.clicked.connect(self.rob.out_of_way)

        layout = QVBoxLayout()
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_input)
        layout.addWidget(self.path_label)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.image_label)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.home_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Save Directory")
        if folder:
            self.save_path = folder
            self.path_label.setText(f"Save Path: {self.save_path}")

    def start_scan(self):
        if not self.scan_in_progress:
            self.scan_in_progress = True
            self.scan_name = self.name_input.text().strip()
            if not self.scan_name:
                print("Please enter a valid scan name.")
                self.scan_in_progress = False
                return

            self.progress_bar.setValue(0)
            self.scan_thread = ScanThread(
                self.rob, self.cam, Z_RANGE, Z_STEP, self.scan_name, self.save_path
            )
            self.scan_thread.progress_updated.connect(self.update_progress)
            self.scan_thread.scan_complete.connect(self.display_best_image)
            self.scan_thread.start()
            print(f"Scan started: {self.scan_name}")
        else:
            print("Scan is already in progress.")

    def display_best_image(self, image):
        self.scan_in_progress = False
        if image is not None:
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_img = QImage(
                image.data, width, height, bytes_per_line, QImage.Format_RGB888
            ).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img).scaled(self.image_label.size())
            self.image_label.setPixmap(pixmap)
            print("Best image displayed.")
        else:
            self.image_label.setText("No image captured.")
            print("No best image to display.")

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

    def closeEvent(self, event):
        self.rob.out_of_way()
        self.scope.end()
        self.cam._camera.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = ScanUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
