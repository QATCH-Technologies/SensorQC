from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QTextEdit,
    QInputDialog,
    QProgressBar,
    QMenuBar,
    QAction,
    QLineEdit,
)
from PyQt5.QtCore import pyqtSignal, QObject, QThread
import sys
import serial.tools.list_ports
import cv2
from robot import Robot
from constants import SystemConstants
import os
import math
import time
import numpy as np
import easyocr


class CaptureSignal(QObject):
    """This class emits a signal when a tile is captured."""

    tile_captured = pyqtSignal(int, int)


class ScanThread(QThread):
    tile_captured = pyqtSignal(int, int)
    scan_complete = pyqtSignal()
    log_message = pyqtSignal(str)
    progress_update = pyqtSignal(int)

    def __init__(self, ui, start_index=0):
        super().__init__()
        self.ui = ui
        self.scanning = False
        self.start_index = start_index

    def run(self):
        self.scanning = True
        total_positions = self.ui.num_tiles_x * self.ui.num_tiles_y
        self.progress_update.emit(self.start_index)

        base_dir = os.path.join(SystemConstants.SERVER_PATH, self.ui.scan_name)
        os.makedirs(base_dir, exist_ok=True)

        total_tiles = self.start_index
        for row in range(self.ui.num_tiles_y):
            col_range = (
                range(self.ui.num_tiles_x)
                if row % 2 == 0
                else range(self.ui.num_tiles_x - 1, -1, -1)
            )

            for col in col_range:
                current_index = row * self.ui.num_tiles_x + col
                if current_index < self.start_index:
                    continue

                if not self.scanning:
                    self.ui.scan_index = current_index
                    self.log_message.emit(
                        f"Scan paused at index {self.ui.scan_index}.")
                    return

                x, y = self.ui.tile_positions[(row, col)]
                self.log_message.emit(
                    f"Moving to ({x:.2f}, {y:.2f}) and capturing image at ({row}, {col})"
                )
                self.ui.robot.go_to(
                    x_position=x, y_position=y, z_position=SystemConstants.Z_HEIGHT)

                time.sleep(SystemConstants.TILE_TO_TILE_DELAY)
                if self.ui.cap is not None:
                    ret, frame = self.ui.cap.read()
                    if ret:
                        self.ui.tile_images[(row, col)] = frame
                        self.ui.row_images.append(frame)
                    else:
                        self.log_message.emit(
                            f"Error capturing image at ({row}, {col})"
                        )

                self.tile_captured.emit(row, col)
                total_tiles += 1
                self.progress_update.emit(total_tiles)

            for col in range(self.ui.num_tiles_x):
                adj_col = (self.ui.num_tiles_x - 1 -
                           col) if row % 2 != 0 else col
                image_path = os.path.join(
                    base_dir, f"tile_{row}_{adj_col}.jpg")
                image = self.ui.row_images[col]
                if image is not None:
                    cv2.imwrite(image_path, image)
            self.log_message.emit(f"Writing row {row}")
            self.ui.row_images = []

        self.scanning = False
        self.log_message.emit("Scanning complete.")
        self.scan_complete.emit()

    def stop(self):
        self.scanning = False


class ScanUI(QWidget):
    def __init__(
        self,
        top_left,
        bottom_right,
        tile_size=(SystemConstants.X_DELTA, SystemConstants.Y_DELTA),
    ):
        super().__init__()
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.tile_width, self.tile_height = tile_size
        self.rect_width = abs(bottom_right[0] - top_left[0])
        self.rect_height = abs(bottom_right[1] - top_left[1])
        self.num_tiles_x = int(math.ceil(self.rect_width / self.tile_width))
        self.num_tiles_y = int(math.ceil(self.rect_height / self.tile_height))
        self.tiles = {}
        self.tile_positions = {}
        self.scanning = False
        self.scan_index = 0
        self.scan_name = None
        self.selected_serial_port = None
        self.selected_camera = None
        self.resume_index = 0
        self.scan_thread = None
        self.robot = None
        self.cap = None
        self.sesnor_id = ""
        self.row_images = []
        self.tile_images = {}
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        # Menu bar
        self.menubar = QMenuBar(self)
        self.settings_menu = self.menubar.addMenu("Settings")

        # Serial port selection action
        self.serial_action = QAction("Select Serial Port", self)
        self.serial_action.triggered.connect(self.select_serial_port)
        self.settings_menu.addAction(self.serial_action)

        # Camera selection action
        self.camera_action = QAction("Select Camera", self)
        self.camera_action.triggered.connect(self.select_camera)
        self.settings_menu.addAction(self.camera_action)

        self.layout.setMenuBar(self.menubar)

        # Grid layout for the rectangles
        self.grid_layout = QGridLayout()
        self.layout.addLayout(self.grid_layout)

        # Button layout
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("New Scan")
        self.stop_button = QPushButton("Stop")
        self.reset_button = QPushButton("Reset")
        self.resume_button = QPushButton("Resume")

        self.run_button.clicked.connect(self.run_action)
        self.stop_button.clicked.connect(self.stop_action)
        self.reset_button.clicked.connect(self.reset_action)
        self.resume_button.clicked.connect(self.resume_action)

        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.resume_button)
        self.layout.addLayout(button_layout)
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)

        # Console output
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.layout.addWidget(self.console_output)

        self.setLayout(self.layout)
        self.setWindowTitle("Scanner")

        self.generate_grid()
        self.show()

    def generate_grid(self):
        """Generate and display the grid with physical positions."""
        self.log_to_console("Generating grid...")

        for i in range(self.num_tiles_y):
            for j in range(self.num_tiles_x):
                label = QLabel(f"{i},{j}")
                label.setStyleSheet(
                    "border: 1px solid black; background-color: white; padding: 5px;"
                )
                self.grid_layout.addWidget(label, i, j)
                self.tiles[(i, j)] = label

                # Calculate physical coordinates
                x_pos = self.top_left[0] + (j * self.tile_width)
                # Y decreases as rows go down
                y_pos = self.top_left[1] - (i * self.tile_height)
                self.tile_positions[(i, j)] = (x_pos, y_pos)
        self.log_to_console(
            f"Grid generated with {self.num_tiles_x} x {self.num_tiles_y} cells."
        )

    def reset_action(self):
        self.stop_action()
        self.scanning = False
        self.scan_index = 0
        self.progress_bar.setValue(0)
        self.scan_name = None
        self.robot.out_of_way()
        for label in self.tiles.values():
            label.setStyleSheet(
                "border: 1px solid black; background-color: white; padding: 5px;"
            )
        self.log_to_console(
            "Scan reset: Grid cleared and ready for a new run.")

    def scan_complete(self):
        self.log_to_console("Scan process completed.")

    def run_action(self):
        if self.robot is None:
            self.log_to_console(
                "Error: No robot selected. Please select a serial port before running."
            )
            return

        if self.cap is None:
            self.log_to_console(
                "Error: No camera selected. Please select a camera port before running."
            )
            return

        status, id_frame = self.cap.read()
        if status:
            self.get_sensor_id(id_frame)
            default_scan_name = "Sensor_" + self.sensor_id
        else:
            self.log_to_console("Error: Could not perform OCR for sensor ID.")
            default_scan_name = "Sensor_"

        scan_name, ok = QInputDialog.getText(
            self, "Scan Name", "Enter scan name:", QLineEdit.Normal, default_scan_name
        )
        if ok and scan_name.strip():
            self.scan_name = scan_name
            self.log_to_console(f"Scan started with name: {scan_name}")
            self.start_scan_thread()
        else:
            self.log_to_console("Scan canceled or invalid name entered.")

    def start_scan_thread(self, start_index=0):
        self.scan_thread = ScanThread(self, start_index=start_index)
        self.scan_thread.tile_captured.connect(self.update_tile_color)
        self.scan_thread.log_message.connect(self.log_to_console)
        self.scan_thread.progress_update.connect(self.progress_bar.setValue)
        self.scan_thread.scan_complete.connect(self.scan_complete)
        self.scan_thread.start()

    def resume_action(self):
        if self.scan_name and self.scan_index > 0:
            self.log_to_console(f"Resuming scan from index {self.scan_index}.")
            self.start_scan_thread(start_index=self.scan_index)
        else:
            self.log_to_console("No previous scan to resume.")
            self.run_action()

    def stop_action(self):
        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.stop()
            self.log_to_console("Scan stopped.")

        self.scanning = False  # This will cause scan_grid to exit its loop
        self.log_to_console(f"Scan paused at index {self.scan_index}.")

    def scan_complete(self):
        self.log_to_console("Scan process completed.")

    def closeEvent(self, event):
        """Ensure resources are released on window close."""
        self.log_to_console("Closing application...")

        # Release the camera if it's open
        if self.selected_camera is not None:
            cap = cv2.VideoCapture(self.selected_camera)
            if cap.isOpened():
                cap.release()
                self.log_to_console("Camera released.")

        # End the robot connection
        if self.robot is not None:
            self.robot.end()
            self.log_to_console("Robot connection closed.")

        event.accept()

    def update_tile_color(self, row, col):
        if (row, col) in self.tiles:
            self.tiles[(row, col)].setStyleSheet(
                "border: 1px solid black; background-color: green; padding: 5px;"
            )
            self.log_to_console(f"Tile at ({row},{col}) captured.")

    def select_serial_port(self):
        """Opens a dialog to select a serial port."""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        if not ports:
            self.log_to_console("No serial ports available.")
            return

        port, ok = QInputDialog.getItem(
            self, "Select Serial Port", "Available Ports:", ports, 0, False
        )

        if ok and port:
            self.selected_serial_port = port
            self.log_to_console(
                f"Selected Serial Port: {self.selected_serial_port}")
            self.robot = Robot(port=self.selected_serial_port)
            self.robot.begin()
            self.robot.go_to(
                SystemConstants.TOP_LEFT[0], SystemConstants.TOP_LEFT[1], SystemConstants.Z_HEIGHT)

    def select_camera(self):
        """Opens a dialog to select a camera."""
        cameras = []
        for i in range(SystemConstants.NUM_VIDEO_CAPTURE_DEVICES):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append(f"Camera {i}")
                cap.release()

        if not cameras:
            self.log_to_console("No cameras available.")
            return

        camera, ok = QInputDialog.getItem(
            self, "Select Camera", "Available Cameras:", cameras, 0, False
        )

        if ok and camera:
            self.selected_camera = int(camera.split()[-1])
            self.log_to_console(f"Selected Camera: {self.selected_camera}")
            self.cap = cv2.VideoCapture(self.selected_camera)

    def get_sensor_id(self, frame):
        self.log_to_console("Performing OCR on frame.")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, None, fx=2, fy=2,
                             interpolation=cv2.INTER_LINEAR)

        reader = easyocr.Reader(["en"])
        result = reader.readtext(np.array(resized))
        id_string = result[1][0]
        if id_string[0] == "1":
            id_string[0] = "I"
        self.sensor_id = id_string

    def compute_tile_dimensions(self, z_height):
        z_values = np.array([6.5, 45.0])
        width_values = np.array([1.4, 14.7])
        height_values = np.array([1.0, 7.9])

        width_coeffs = np.polyfit(np.log(z_values), np.log(width_values), 1)
        height_coeffs = np.polyfit(np.log(z_values), np.log(height_values), 1)

        predicted_width = np.exp(width_coeffs[1]) * z_height**width_coeffs[0]
        predicted_height = np.exp(
            height_coeffs[1]) * z_height**height_coeffs[0]

        return predicted_width, predicted_height

    def log_to_console(self, message):
        """Logs messages to the embedded console output."""
        self.console_output.append(message)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    ex = ScanUI(SystemConstants.TOP_LEFT, SystemConstants.BOTTOM_RIGHT)
    sys.exit(app.exec_())
