import sys
import os
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QPushButton, QLabel, QLineEdit, QProgressBar, QFileDialog, QSizePolicy,
    QTabWidget, QGroupBox, QSplitter, QDoubleSpinBox, QStyle, QSpacerItem
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont

# Import your custom modules.
from dino_lite_edge import Camera, Microscope
from robot import Robot
from constants import MicroscopeConstants

# Default values
DEFAULT_INLET_POSITION = (111.9, 125.2, 10.7)
DEFAULT_Z_RANGE = (10.7, 11.0)
DEFAULT_Z_STEP = 0.05


def compute_offset(target_img, live_img):
    """
    Compute the (x,y) offset between a target image and a live image using ORB features.
    """
    gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    gray_live = cv2.cvtColor(live_img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray_target, None)
    kp2, des2 = orb.detectAndCompute(gray_live, None)
    if des1 is None or des2 is None:
        return None, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < 4:
        return None, None
    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is not None:
        offset_x = M[0, 2]
        offset_y = M[1, 2]
        return offset_x, offset_y
    return None, None


def apply_scale(image):
    """
    Draw a grid and scale labels on the image.
    """
    height, width, _ = image.shape
    major_x_spacing = 304  # 0.1 mm in X
    major_y_spacing = 324  # 0.1 mm in Y
    minor_x_spacing = major_x_spacing // 2  # 0.05 mm in X
    minor_y_spacing = major_y_spacing // 2  # 0.05 mm in Y
    major_grid_color = (235, 90, 60)
    minor_grid_color = (223, 151, 85)
    text_color = (235, 90, 60)
    thickness_major = 2
    thickness_minor = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    for x in range(0, width, minor_x_spacing):
        cv2.line(image, (x, 0), (x, height), minor_grid_color, thickness_minor)
    for y in range(0, height, minor_y_spacing):
        cv2.line(image, (0, y), (width, y), minor_grid_color, thickness_minor)
    for x in range(0, width, major_x_spacing):
        cv2.line(image, (x, 0), (x, height), major_grid_color, thickness_major)
    for y in range(0, height, major_y_spacing):
        cv2.line(image, (0, y), (width, y), major_grid_color, thickness_major)
    for x in range(0, width, major_x_spacing):
        for y in range(0, height, major_y_spacing):
            label = f"({x//major_x_spacing * 0.1:.1f}, {y//major_y_spacing * 0.1:.1f}) mm"
            cv2.putText(image, label, (x + 5, y + 15),
                        font, font_scale, text_color, 1)
    return image


class CameraThread(QThread):
    frame_captured = pyqtSignal(np.ndarray)

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.running = True

    def run(self):
        while self.running:
            status, frame = self.camera._camera.read()
            if status:
                self.frame_captured.emit(frame)
            self.msleep(1)

    def stop(self):
        self.running = False
        self.wait()


class ScanThread(QThread):
    progress_updated = pyqtSignal(int)
    scan_complete = pyqtSignal(np.ndarray)

    def __init__(self, rob: Robot, cam: Camera, z_range, z_step, scan_name, save_path, inlet_position):
        super().__init__()
        self.rob = rob
        self.cam = cam
        self.z_range = z_range
        self.z_step = z_step
        self.scan_name = scan_name
        self.save_path = save_path
        self.inlet_position = inlet_position
        self.best_image = None
        self.best_sharpness = -np.inf

    def run(self):
        total_steps = len(
            np.arange(self.z_range[0], self.z_range[1], self.z_step))
        current_step = 0
        # Move to the inlet position.
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
            return apply_scale(straightened)
        else:
            print("No contours found, skipping straightening.")
            return apply_scale(self.best_image)

    def calculate_sharpness(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        return laplacian.var()


class ScanUI(QWidget):
    """
    Central widget for scanning, with a clean, minimal interface.
    Status messages are forwarded to the parent (QMainWindow) if available.
    """

    def __init__(self):
        super().__init__()
        self.scan_name = "Sensor_"
        self.save_path = os.getcwd()
        self.scan_in_progress = False
        self.cam = Camera()
        self.scope = Microscope()
        self.rob = Robot()
        self.scope.led_on(MicroscopeConstants.DARK_FIELD)
        self.rob.begin()
        self.current_pos = list(DEFAULT_INLET_POSITION)
        self.x_step = 0.1
        self.y_step = 0.1
        self.z_step = DEFAULT_Z_STEP
        self.init_ui()
        self.init_camera_thread()

    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #FFFFFF; }
            QWidget {
                background-color: #FDFDFD;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-size: 14px;
            }
            QGroupBox {
                border: none;
                margin-top: 20px;
                padding: 10px;
            }
            QGroupBox::title {
                color: #888;
                font-size: 12px;
                padding: 0 5px;
            }
            /* Use QATCH_BLUE_FG (#00A3DA) for button background
               and QATCH_GREY_BG (#F6F6F6) for button text */
            QPushButton {
                background-color: #00A3DA;
                color: #F6F6F6;
                border: none;
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #0081B0;
            }
            QDoubleSpinBox, QLineEdit {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 4px 8px;
                background-color: #FFF;
            }
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 6px;
                background-color: #EEE;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #00A3DA;
                border-radius: 6px;
            }
            QTabWidget::pane {
                border: none;
                background: #FFF;
            }
            QTabBar::tab {
                background: #F7F7F7;
                padding: 8px 15px;
                margin-right: 2px;
                font-weight: 500;
                qproperty-iconSize: 24px 24px;
                min-width: 80px;
                text-align: center;
            }
            QTabBar::tab:selected {
                border-bottom: 2px solid #00A3DA;
                color: #00A3DA;
                background-color: #FFF;
            }
            QTabBar::tab:hover {
                background-color: #F0F0F0;
            }
        """)
        self.tabs = QTabWidget()
        # Set icon size for tabs
        self.tabs.setIconSize(QSize(24, 24))
        self.tabs.addTab(self.create_scan_tab(), "Scan")
        self.tabs.addTab(self.create_control_tab(), "Manual Control")
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def update_status(self, message):
        # Forward status messages to the QMainWindow's status bar if available.
        if self.parent() and hasattr(self.parent(), "statusBar"):
            self.parent().statusBar().showMessage(message, 5000)

    def create_scan_tab(self):
        scan_tab = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        # Scan Info Section
        info_group = QGroupBox("Scan Info")
        form_layout = QFormLayout()
        self.batch_input = QLineEdit()
        self.batch_input.setPlaceholderText("Enter batch name")
        self.batch_input.setFixedWidth(200)
        self.batch_input.setToolTip("Type in the batch name.")
        self.name_input = QLineEdit("Sensor_")
        self.name_input.setFixedWidth(200)
        self.name_input.setToolTip("Type in the sensor name.")
        form_layout.addRow("Batch Name:", self.batch_input)
        form_layout.addRow("Sensor Name:", self.name_input)
        info_group.setLayout(form_layout)
        main_layout.addWidget(info_group)
        # Save Path Section
        path_layout = QHBoxLayout()
        self.path_label = QLabel(f"Save Path: {self.save_path}")
        self.browse_button = QPushButton("Browse")
        self.browse_button.setToolTip("Select folder to save scans.")
        self.browse_button.setIcon(
            self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.browse_button.clicked.connect(self.browse_folder)
        path_layout.addWidget(self.path_label)
        path_layout.addWidget(self.browse_button)
        main_layout.addLayout(path_layout)
        # Inlet and Z Settings Section
        settings_group = QGroupBox("Inlet and Z Settings")
        settings_layout = QGridLayout()
        settings_layout.setSpacing(10)
        self.inlet_x_input = QDoubleSpinBox()
        self.inlet_x_input.setRange(-1000, 1000)
        self.inlet_x_input.setDecimals(2)
        self.inlet_x_input.setValue(DEFAULT_INLET_POSITION[0])
        self.inlet_x_input.setToolTip("Set the X coordinate of the inlet.")
        self.inlet_y_input = QDoubleSpinBox()
        self.inlet_y_input.setRange(-1000, 1000)
        self.inlet_y_input.setDecimals(2)
        self.inlet_y_input.setValue(DEFAULT_INLET_POSITION[1])
        self.inlet_y_input.setToolTip("Set the Y coordinate of the inlet.")
        self.z_range_low_input = QDoubleSpinBox()
        self.z_range_low_input.setRange(0, 100)
        self.z_range_low_input.setDecimals(2)
        self.z_range_low_input.setValue(DEFAULT_Z_RANGE[0])
        self.z_range_low_input.setToolTip("Set the lower Z value.")
        self.z_range_high_input = QDoubleSpinBox()
        self.z_range_high_input.setRange(0, 100)
        self.z_range_high_input.setDecimals(2)
        self.z_range_high_input.setValue(DEFAULT_Z_RANGE[1])
        self.z_range_high_input.setToolTip("Set the higher Z value.")
        self.z_step_input = QDoubleSpinBox()
        self.z_step_input.setRange(0.01, 10)
        self.z_step_input.setDecimals(2)
        self.z_step_input.setValue(DEFAULT_Z_STEP)
        self.z_step_input.setToolTip("Set the Z movement step size.")
        settings_layout.addWidget(QLabel("Inlet X:"), 0, 0)
        settings_layout.addWidget(self.inlet_x_input, 0, 1)
        settings_layout.addWidget(QLabel("Inlet Y:"), 0, 2)
        settings_layout.addWidget(self.inlet_y_input, 0, 3)
        settings_layout.addWidget(QLabel("Z Range Low:"), 1, 0)
        settings_layout.addWidget(self.z_range_low_input, 1, 1)
        settings_layout.addWidget(QLabel("Z Range High:"), 1, 2)
        settings_layout.addWidget(self.z_range_high_input, 1, 3)
        settings_layout.addWidget(QLabel("Z Step:"), 2, 0)
        settings_layout.addWidget(self.z_step_input, 2, 1)
        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)
        # Action Buttons
        btn_layout = QHBoxLayout()
        self.move_location_button = QPushButton("Go to Specified Location")
        self.move_location_button.setToolTip(
            "Move the robot to the specified inlet position.")
        self.move_location_button.setIcon(
            self.style().standardIcon(QStyle.SP_ArrowForward))
        self.move_location_button.clicked.connect(
            self.move_to_specified_location)
        self.visual_align_button = QPushButton("Visual Align")
        self.visual_align_button.setToolTip(
            "Use a target image to visually align the robot.")
        self.visual_align_button.setIcon(
            self.style().standardIcon(QStyle.SP_FileDialogStart))
        self.visual_align_button.clicked.connect(self.visual_align)
        btn_layout.addWidget(self.move_location_button)
        btn_layout.addWidget(self.visual_align_button)
        main_layout.addLayout(btn_layout)
        # Progress Bar and Scan Control Buttons
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        ctrl_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Scan")
        self.start_button.setToolTip("Begin the scanning process.")
        self.start_button.setIcon(
            self.style().standardIcon(QStyle.SP_MediaPlay))
        self.start_button.clicked.connect(self.start_scan)
        self.stop_button = QPushButton("Stop Scan")
        self.stop_button.setToolTip("Stop the current scan.")
        self.stop_button.setIcon(
            self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.clicked.connect(self.stop_scan)
        self.reset_button = QPushButton("Reset Scan")
        self.reset_button.setToolTip(
            "Reset the scan settings and robot position.")
        self.reset_button.setIcon(
            self.style().standardIcon(QStyle.SP_BrowserReload))
        self.reset_button.clicked.connect(self.reset_scan)
        self.home_button = QPushButton("Home")
        self.home_button.setToolTip(
            "Move the robot to the home position (out of way).")
        self.home_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowUp))
        self.home_button.clicked.connect(self.rob.out_of_way)
        ctrl_layout.addWidget(self.start_button)
        ctrl_layout.addWidget(self.stop_button)
        ctrl_layout.addWidget(self.reset_button)
        ctrl_layout.addWidget(self.home_button)
        main_layout.addLayout(ctrl_layout)
        scan_tab.setLayout(main_layout)
        return scan_tab

    def create_control_tab(self):
        control_tab = QWidget()
        splitter = QSplitter(Qt.Horizontal)
        # Left: Live Feed and Position
        live_feed_group = QGroupBox("Live Feed")
        left_layout = QVBoxLayout()
        self.live_feed_label = QLabel("Live Camera Feed")
        self.live_feed_label.setAlignment(Qt.AlignCenter)
        self.live_feed_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.live_feed_label)
        self.position_label = QLabel(
            "Current Position: " + str(self.current_pos))
        left_layout.addWidget(self.position_label)
        live_feed_group.setLayout(left_layout)
        # Right: Robot Control Pad
        control_pad_group = QGroupBox("Robot Control Pad")
        cp_main_layout = QVBoxLayout()
        # XY Control (3x3 grid with center blank)
        xy_group = QGroupBox("XY Control")
        xy_layout = QGridLayout()
        self.y_plus_button = QPushButton("Y+")
        self.y_plus_button.setToolTip("Move robot in positive Y direction.")
        self.y_minus_button = QPushButton("Y-")
        self.y_minus_button.setToolTip("Move robot in negative Y direction.")
        self.x_minus_button = QPushButton("X-")
        self.x_minus_button.setToolTip("Move robot in negative X direction.")
        self.x_plus_button = QPushButton("X+")
        self.x_plus_button.setToolTip("Move robot in positive X direction.")
        xy_layout.addWidget(self.y_plus_button, 0, 1)
        xy_layout.addWidget(self.x_minus_button, 1, 0)
        xy_layout.addWidget(self.x_plus_button, 1, 2)
        xy_layout.addWidget(self.y_minus_button, 2, 1)
        xy_group.setLayout(xy_layout)
        # Z Control
        z_group = QGroupBox("Z Control")
        z_layout = QVBoxLayout()
        self.z_plus_button = QPushButton("Z+")
        self.z_plus_button.setToolTip("Move robot upward (positive Z).")
        self.z_minus_button = QPushButton("Z-")
        self.z_minus_button.setToolTip("Move robot downward (negative Z).")
        z_layout.addWidget(self.z_plus_button)
        z_layout.addWidget(self.z_minus_button)
        z_group.setLayout(z_layout)
        cp_main_layout.addWidget(xy_group)
        cp_main_layout.addWidget(z_group)
        control_pad_group.setLayout(cp_main_layout)
        # Connect control pad buttons
        self.x_plus_button.clicked.connect(self.move_x_positive)
        self.x_minus_button.clicked.connect(self.move_x_negative)
        self.y_plus_button.clicked.connect(self.move_y_positive)
        self.y_minus_button.clicked.connect(self.move_y_negative)
        self.z_plus_button.clicked.connect(self.move_z_positive)
        self.z_minus_button.clicked.connect(self.move_z_negative)
        splitter.addWidget(live_feed_group)
        splitter.addWidget(control_pad_group)
        splitter.setSizes([700, 300])
        layout = QVBoxLayout()
        layout.addWidget(splitter)
        control_tab.setLayout(layout)
        return control_tab

    def init_camera_thread(self):
        self.camera_thread = CameraThread(self.cam)
        self.camera_thread.frame_captured.connect(self.update_live_feed_thread)
        self.camera_thread.start()

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Save Directory")
        if folder:
            self.save_path = folder
            self.path_label.setText(f"Save Path: {self.save_path}")
            self.update_status("Save path updated.")

    def move_to_specified_location(self):
        inlet_x = self.inlet_x_input.value()
        inlet_y = self.inlet_y_input.value()
        z_low = self.z_range_low_input.value()
        inlet_position = (inlet_x, inlet_y, z_low)
        self.current_pos = list(inlet_position)
        self.rob.go_to(inlet_x, inlet_y, z_low)
        self.update_position_label()
        self.update_status(f"Moved to location: {inlet_position}")

    def start_scan(self):
        if not self.scan_in_progress:
            self.scan_in_progress = True
            sensor_name = self.name_input.text().strip()
            batch_name = self.batch_input.text().strip()
            self.scan_name = f"{batch_name}_{sensor_name}" if batch_name else sensor_name
            if not self.scan_name:
                self.update_status("Please enter a valid scan name.")
                self.scan_in_progress = False
                return
            inlet_x = self.inlet_x_input.value()
            inlet_y = self.inlet_y_input.value()
            z_low = self.z_range_low_input.value()
            z_high = self.z_range_high_input.value()
            z_step = self.z_step_input.value()
            inlet_position = (inlet_x, inlet_y, z_low)
            z_range = (z_low, z_high)
            self.current_pos = list(inlet_position)
            self.z_step = z_step
            self.progress_bar.setValue(0)
            self.scan_thread = ScanThread(self.rob, self.cam, z_range, z_step,
                                          self.scan_name, self.save_path, inlet_position)
            self.scan_thread.progress_updated.connect(self.update_progress)
            self.scan_thread.scan_complete.connect(self.display_best_image)
            self.scan_thread.start()
            self.update_status(f"Scan started: {self.scan_name}")
        else:
            self.update_status("Scan is already in progress.")

    def display_best_image(self, image):
        self.scan_in_progress = False
        if image is not None:
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_img = QImage(image.data, width, height,
                           bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img).scaled(
                self.live_feed_label.size(), Qt.KeepAspectRatio)
            self.live_feed_label.setPixmap(pixmap)
            self.update_status("Best image captured and displayed.")
        else:
            self.live_feed_label.setText("No image captured.")
            self.update_status("No best image to display.")

    def stop_scan(self):
        if self.scan_in_progress:
            self.rob.out_of_way()
            self.scan_thread.terminate()
            self.scan_in_progress = False
            self.progress_bar.setValue(0)
            self.update_status(f"Scan stopped: {self.scan_name}")
        else:
            self.update_status("No scan in progress.")

    def reset_scan(self):
        if self.scan_in_progress:
            self.scan_thread.terminate()
            self.scan_in_progress = False
        inlet_x = self.inlet_x_input.value()
        inlet_y = self.inlet_y_input.value()
        z_low = self.z_range_low_input.value()
        inlet_position = (inlet_x, inlet_y, z_low)
        self.current_pos = list(inlet_position)
        self.rob.go_to(inlet_position[0], inlet_position[1], inlet_position[2])
        time.sleep(1)
        self.name_input.setText("Sensor_")
        self.progress_bar.setValue(0)
        self.update_status("Scan reset.")

    def update_progress(self, progress):
        self.progress_bar.setValue(progress)

    def move_x_positive(self):
        self.current_pos[0] += self.x_step
        self.rob.go_to(self.current_pos[0],
                       self.current_pos[1], self.current_pos[2])
        self.update_position_label()

    def move_x_negative(self):
        self.current_pos[0] -= self.x_step
        self.rob.go_to(self.current_pos[0],
                       self.current_pos[1], self.current_pos[2])
        self.update_position_label()

    def move_y_positive(self):
        self.current_pos[1] += self.y_step
        self.rob.go_to(self.current_pos[0],
                       self.current_pos[1], self.current_pos[2])
        self.update_position_label()

    def move_y_negative(self):
        self.current_pos[1] -= self.y_step
        self.rob.go_to(self.current_pos[0],
                       self.current_pos[1], self.current_pos[2])
        self.update_position_label()

    def move_z_positive(self):
        self.current_pos[2] += self.z_step
        self.rob.go_to(self.current_pos[0],
                       self.current_pos[1], self.current_pos[2])
        self.update_position_label()

    def move_z_negative(self):
        self.current_pos[2] -= self.z_step
        self.rob.go_to(self.current_pos[0],
                       self.current_pos[1], self.current_pos[2])
        self.update_position_label()

    def update_position_label(self):
        self.position_label.setText(
            "Current Position: " + str(self.current_pos))

    def visual_align(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Target Image", "", "Images (*.png *.jpg *.bmp)")
        if not file_name:
            return
        target_img = cv2.imread(file_name)
        if target_img is None:
            self.update_status("Could not load target image.")
            return
        pixel_to_mm = 0.1
        tolerance = 2.0
        max_iterations = 10
        iteration = 0
        self.update_status("Starting visual alignment...")
        while iteration < max_iterations:
            status, live_img = self.cam._camera.read()
            if not status:
                self.update_status("Failed to capture live image. Retrying...")
                time.sleep(1)
                continue
            offset = compute_offset(target_img, live_img)
            if offset == (None, None):
                self.update_status(
                    "Insufficient feature matches. Trying again...")
                time.sleep(1)
                iteration += 1
                continue
            offset_x, offset_y = offset
            if abs(offset_x) < tolerance and abs(offset_y) < tolerance:
                self.update_status(
                    "Alignment within tolerance. Alignment complete!")
                break
            new_x = self.current_pos[0] + offset_x * pixel_to_mm
            new_y = self.current_pos[1] + offset_y * pixel_to_mm
            self.rob.go_to(new_x, new_y, self.current_pos[2])
            self.current_pos[0] = new_x
            self.current_pos[1] = new_y
            time.sleep(1)
            iteration += 1

    def update_live_feed_thread(self, frame):
        scaled_frame = apply_scale(frame.copy())
        frame_rgb = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame_rgb.data, width, height,
                       bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.live_feed_label.size(), Qt.KeepAspectRatio)
        self.live_feed_label.setPixmap(pixmap)
        self.update_position_label()

    def keyPressEvent(self, event):
        if self.tabs.currentIndex() == 1:
            key = event.key()
            if key == Qt.Key_W:
                self.move_y_positive()
            elif key == Qt.Key_S:
                self.move_y_negative()
            elif key == Qt.Key_A:
                self.move_x_negative()
            elif key == Qt.Key_D:
                self.move_x_positive()
            elif key == Qt.Key_Up:
                self.move_z_positive()
            elif key == Qt.Key_Down:
                self.move_z_negative()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.camera_thread.stop()
        self.rob.out_of_way()
        self.scope.end()
        self.cam._camera.release()
        event.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scan Control Panel")
        self.resize(980, 750)
        self.scan_ui = ScanUI()
        self.setCentralWidget(self.scan_ui)
        # Use the built-in status bar for professional feedback.
        self.statusBar().showMessage("Ready")
        # Set icons for the tabs.
        self.scan_ui.tabs.setTabIcon(
            0, self.style().standardIcon(QStyle.SP_FileIcon))
        self.scan_ui.tabs.setTabIcon(
            1, self.style().standardIcon(QStyle.SP_ComputerIcon))


def main():
    app = QApplication(sys.argv)
    # Use the Fusion style for a consistent look.
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
