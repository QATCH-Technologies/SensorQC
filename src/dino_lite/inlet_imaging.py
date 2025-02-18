import sys
import os
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QProgressBar,
    QFileDialog,
    QSizePolicy
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage
import logging as log

from dino_lite_edge import Camera, Microscope
from robot import Robot
from constants import MicroscopeConstants

# Default values
DEFAULT_INLET_POSITION = (111.9, 125.2, 10.7)
DEFAULT_Z_RANGE = (10.7, 11)
DEFAULT_Z_STEP = 0.05


def compute_offset(target_img, live_img):
    """
    Given a target image and a live image, compute the (x, y) offset in pixels using feature matching.
    Returns (offset_x, offset_y) or (None, None) if insufficient matches.
    """
    # Convert images to grayscale
    gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    gray_live = cv2.cvtColor(live_img, cv2.COLOR_BGR2GRAY)

    # Detect features using ORB
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray_target, None)
    kp2, des2 = orb.detectAndCompute(gray_live, None)

    if des1 is None or des2 is None:
        return None, None

    # Match features using a brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 4:
        # Not enough matches to compute a reliable transformation
        return None, None

    # Extract matched keypoints
    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography using RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is not None:
        offset_x = M[0, 2]
        offset_y = M[1, 2]
        return offset_x, offset_y
    return None, None


class ScanThread(QThread):
    progress_updated = pyqtSignal(int)
    scan_complete = pyqtSignal(np.ndarray)

    def __init__(self, rob: Robot, cam: Camera, z_range, z_step, scan_name, save_path, inlet_position):
        super().__init__()
        self.rob = rob
        self.cam = cam
        self.z_range = z_range      # Tuple: (z_min, z_max)
        self.z_step = z_step
        self.scan_name = scan_name
        self.save_path = save_path
        # Tuple: (inlet_x, inlet_y, z_min)
        self.inlet_position = inlet_position
        self.best_image = None
        self.best_sharpness = -np.inf

    def run(self):
        total_steps = len(
            np.arange(self.z_range[0], self.z_range[1], self.z_step))
        current_step = 0

        # Move to the inlet position using the user-defined values.
        self.rob.go_to(
            self.inlet_position[0], self.inlet_position[1], self.inlet_position[2])
        time.sleep(4)

        for z in np.arange(self.z_range[0], self.z_range[1], self.z_step):
            self.rob.go_to(self.inlet_position[0], self.inlet_position[1], z)
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
        return image

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

            (h, w) = self.best_image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            straightened = cv2.warpAffine(self.best_image, rotation_matrix, (w, h),
                                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return self.apply_scale(straightened)
        else:
            print("No contours found, skipping straightening.")
            return self.apply_scale(self.best_image)

    def calculate_sharpness(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        return laplacian.var()


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

        # Set the initial current position using default inlet values.
        self.current_pos = list(DEFAULT_INLET_POSITION)
        # Define step sizes for manual control (adjust as needed)
        self.x_step = 0.1
        self.y_step = 0.1
        self.z_step = DEFAULT_Z_STEP

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Scan Control Panel")
        self.resize(600, 800)  # Initial window size

        main_layout = QVBoxLayout()

        # --- Top Section: Scan Controls ---
        self.name_label = QLabel("Scan Name:")
        self.name_input = QLineEdit(self)
        self.name_input.setText(self.scan_name)

        self.path_label = QLabel(f"Save Path: {self.save_path}")
        self.browse_button = QPushButton("Browse", self)
        self.browse_button.clicked.connect(self.browse_folder)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        # --- New Settings Section: Inlet and Z Settings ---
        settings_label = QLabel("Inlet and Z Settings")
        settings_layout = QGridLayout()
        self.inlet_x_label = QLabel("Inlet X:")
        self.inlet_x_input = QLineEdit()
        self.inlet_x_input.setText(str(DEFAULT_INLET_POSITION[0]))

        self.inlet_y_label = QLabel("Inlet Y:")
        self.inlet_y_input = QLineEdit()
        self.inlet_y_input.setText(str(DEFAULT_INLET_POSITION[1]))

        self.z_step_label = QLabel("Z Step:")
        self.z_step_input = QLineEdit()
        self.z_step_input.setText(str(DEFAULT_Z_STEP))

        self.z_range_low_label = QLabel("Z Range Low:")
        self.z_range_low_input = QLineEdit()
        self.z_range_low_input.setText(str(DEFAULT_Z_RANGE[0]))

        self.z_range_high_label = QLabel("Z Range High:")
        self.z_range_high_input = QLineEdit()
        self.z_range_high_input.setText(str(DEFAULT_Z_RANGE[1]))

        # Arrange settings in a grid layout.
        settings_layout.addWidget(self.inlet_x_label, 0, 0)
        settings_layout.addWidget(self.inlet_x_input, 0, 1)
        settings_layout.addWidget(self.inlet_y_label, 0, 2)
        settings_layout.addWidget(self.inlet_y_input, 0, 3)
        settings_layout.addWidget(self.z_step_label, 1, 0)
        settings_layout.addWidget(self.z_step_input, 1, 1)
        settings_layout.addWidget(self.z_range_low_label, 1, 2)
        settings_layout.addWidget(self.z_range_low_input, 1, 3)
        settings_layout.addWidget(self.z_range_high_label, 1, 4)
        settings_layout.addWidget(self.z_range_high_input, 1, 5)

        # --- Button: Move to Specified Location ---
        self.move_location_button = QPushButton(
            "Go to Specified Location", self)
        self.move_location_button.clicked.connect(
            self.move_to_specified_location)

        # --- New Button: Visual Align ---
        self.visual_align_button = QPushButton("Auto Align", self)
        self.visual_align_button.clicked.connect(self.auto_align)

        # --- Image Display ---
        self.image_label = QLabel("Best Image will appear here")
        self.image_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setAlignment(Qt.AlignCenter)

        # --- Scan Control Buttons ---
        self.start_button = QPushButton("Start Scan", self)
        self.stop_button = QPushButton("Stop Scan", self)
        self.reset_button = QPushButton("Reset Scan", self)
        self.home_button = QPushButton("Home", self)

        self.start_button.clicked.connect(self.start_scan)
        self.stop_button.clicked.connect(self.stop_scan)
        self.reset_button.clicked.connect(self.reset_scan)
        self.home_button.clicked.connect(self.rob.out_of_way)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addWidget(self.reset_button)
        buttons_layout.addWidget(self.home_button)

        # Add top section widgets to the main layout
        main_layout.addWidget(self.name_label)
        main_layout.addWidget(self.name_input)
        main_layout.addWidget(self.path_label)
        main_layout.addWidget(self.browse_button)
        main_layout.addWidget(settings_label)
        main_layout.addLayout(settings_layout)
        main_layout.addWidget(self.move_location_button)
        # Add the new Visual Align button below the move-to-location button
        main_layout.addWidget(self.visual_align_button)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.image_label, stretch=1)
        main_layout.addLayout(buttons_layout)

        # --- Middle Section: Robot Control Pad ---
        control_pad_label = QLabel("Robot Control Pad")
        main_layout.addWidget(control_pad_label)

        # X and Y control buttons using a grid layout
        control_grid = QGridLayout()
        self.y_plus_button = QPushButton("Y+")
        self.y_minus_button = QPushButton("Y-")
        self.x_minus_button = QPushButton("X-")
        self.x_plus_button = QPushButton("X+")

        control_grid.addWidget(self.y_plus_button, 0, 1)
        control_grid.addWidget(self.x_minus_button, 1, 0)
        control_grid.addWidget(self.x_plus_button, 1, 2)
        control_grid.addWidget(self.y_minus_button, 2, 1)
        main_layout.addLayout(control_grid)

        # Z control buttons arranged horizontally
        z_layout = QHBoxLayout()
        self.z_plus_button = QPushButton("Z+")
        self.z_minus_button = QPushButton("Z-")
        z_layout.addWidget(self.z_plus_button)
        z_layout.addWidget(self.z_minus_button)
        main_layout.addLayout(z_layout)

        # Connect control pad buttons to movement functions
        self.x_plus_button.clicked.connect(self.move_x_positive)
        self.x_minus_button.clicked.connect(self.move_x_negative)
        self.y_plus_button.clicked.connect(self.move_y_positive)
        self.y_minus_button.clicked.connect(self.move_y_negative)
        self.z_plus_button.clicked.connect(self.move_z_positive)
        self.z_minus_button.clicked.connect(self.move_z_negative)

        self.setLayout(main_layout)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Save Directory")
        if folder:
            self.save_path = folder
            self.path_label.setText(f"Save Path: {self.save_path}")

    def move_to_specified_location(self):
        """Reads the inlet and Z settings and commands the robot to move there."""
        try:
            inlet_x = float(self.inlet_x_input.text())
            inlet_y = float(self.inlet_y_input.text())
            # We'll use the lower bound of the Z range as the Z coordinate.
            z_range_low = float(self.z_range_low_input.text())
        except ValueError:
            print("Invalid input in one or more position fields.")
            return

        inlet_position = (inlet_x, inlet_y, z_range_low)
        self.current_pos = list(inlet_position)
        print(f"Moving to specified location: {inlet_position}")
        self.rob.go_to(inlet_x, inlet_y, z_range_low)

    def start_scan(self):
        if not self.scan_in_progress:
            self.scan_in_progress = True
            self.scan_name = self.name_input.text().strip()
            if not self.scan_name:
                print("Please enter a valid scan name.")
                self.scan_in_progress = False
                return

            # Read and parse the inlet and Z settings from the UI.
            try:
                inlet_x = float(self.inlet_x_input.text())
                inlet_y = float(self.inlet_y_input.text())
                z_step = float(self.z_step_input.text())
                z_range_low = float(self.z_range_low_input.text())
                z_range_high = float(self.z_range_high_input.text())
            except ValueError:
                print("Invalid numerical input in settings.")
                self.scan_in_progress = False
                return

            inlet_position = (inlet_x, inlet_y, z_range_low)
            z_range = (z_range_low, z_range_high)
            self.current_pos = list(inlet_position)
            self.z_step = z_step

            self.progress_bar.setValue(0)
            self.scan_thread = ScanThread(self.rob, self.cam, z_range, z_step,
                                          self.scan_name, self.save_path, inlet_position)
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
            q_img = QImage(image.data, width, height,
                           bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img).scaled(
                self.image_label.size(), Qt.KeepAspectRatio)
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
        try:
            inlet_x = float(self.inlet_x_input.text())
            inlet_y = float(self.inlet_y_input.text())
            z_range_low = float(self.z_range_low_input.text())
        except ValueError:
            inlet_x, inlet_y, z_range_low = DEFAULT_INLET_POSITION
        inlet_position = (inlet_x, inlet_y, z_range_low)
        self.current_pos = list(inlet_position)
        self.rob.go_to(inlet_position[0], inlet_position[1], inlet_position[2])
        time.sleep(1)
        self.name_input.setText("Sensor_")
        self.progress_bar.setValue(0)
        print("Scan reset.")

    def update_progress(self, progress):
        self.progress_bar.setValue(progress)

    # --- Robot Control Pad Movement Functions ---
    def move_x_positive(self):
        self.current_pos[0] += self.x_step
        print(f"Moving X+ to {self.current_pos[0]}")
        self.rob.go_to(self.current_pos[0],
                       self.current_pos[1], self.current_pos[2])

    def move_x_negative(self):
        self.current_pos[0] -= self.x_step
        print(f"Moving X- to {self.current_pos[0]}")
        self.rob.go_to(self.current_pos[0],
                       self.current_pos[1], self.current_pos[2])

    def move_y_positive(self):
        self.current_pos[1] += self.y_step
        print(f"Moving Y+ to {self.current_pos[1]}")
        self.rob.go_to(self.current_pos[0],
                       self.current_pos[1], self.current_pos[2])

    def move_y_negative(self):
        self.current_pos[1] -= self.y_step
        print(f"Moving Y- to {self.current_pos[1]}")
        self.rob.go_to(self.current_pos[0],
                       self.current_pos[1], self.current_pos[2])

    def move_z_positive(self):
        self.current_pos[2] += self.z_step
        print(f"Moving Z+ to {self.current_pos[2]}")
        self.rob.go_to(self.current_pos[0],
                       self.current_pos[1], self.current_pos[2])

    def move_z_negative(self):
        self.current_pos[2] -= self.z_step
        print(f"Moving Z- to {self.current_pos[2]}")
        self.rob.go_to(self.current_pos[0],
                       self.current_pos[1], self.current_pos[2])
    # ------------------------------------------------

    def auto_align(self):
        """
        New menu option: Visual alignment.
        Prompts the user for a target image and then continuously adjusts
        the robot's X,Y position based on the offset between the target and the live image.
        """
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Target Image", "", "Images (*.png *.jpg *.bmp)")
        if not file_name:
            return

        target_img = cv2.imread(file_name)
        if target_img is None:
            print("Could not load target image.")
            return

        # Calibration: conversion factor from pixels to robot units (mm)
        pixel_to_mm = 0.1  # Adjust this value after calibration
        tolerance = 2.0    # Tolerance in pixels
        max_iterations = 10
        iteration = 0

        print("Starting visual alignment...")
        while iteration < max_iterations:
            status, live_img = self.cam._camera.read()
            if not status:
                print("Failed to capture live image. Retrying...")
                time.sleep(1)
                continue

            offset = compute_offset(target_img, live_img)
            if offset == (None, None):
                print("Insufficient feature matches. Trying again...")
                time.sleep(1)
                iteration += 1
                continue

            offset_x, offset_y = offset
            print(
                f"Iteration {iteration}: Offset X: {offset_x:.2f}, Offset Y: {offset_y:.2f}")

            # Check if alignment is within tolerance
            if abs(offset_x) < tolerance and abs(offset_y) < tolerance:
                print("Alignment within tolerance. Alignment complete!")
                break

            # Calculate new robot coordinates based on the offset and conversion factor
            new_x = self.current_pos[0] + offset_x * pixel_to_mm
            new_y = self.current_pos[1] + offset_y * pixel_to_mm
            print(f"Moving robot to X: {new_x:.2f}, Y: {new_y:.2f}")
            self.rob.go_to(new_x, new_y, self.current_pos[2])
            self.current_pos[0] = new_x
            self.current_pos[1] = new_y

            # Allow time for the robot to move and the camera to update
            time.sleep(1)
            iteration += 1

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
