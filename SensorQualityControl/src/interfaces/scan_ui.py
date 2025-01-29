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
)
from PyQt5.QtCore import pyqtSignal, QObject, QTimer
import sys
import serial.tools.list_ports
import cv2
from robot import Robot
from constants import SystemConstants
import os
import math


class CaptureSignal(QObject):
    """This class emits a signal when a tile is captured."""

    tile_captured = pyqtSignal(int, int)


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

        # Compute the overall dimensions
        self.rect_width = abs(bottom_right[0] - top_left[0])
        self.rect_height = abs(bottom_right[1] - top_left[1])

        # Compute how many tiles fit
        self.num_tiles_x = int(math.ceil(self.rect_width / self.tile_width))
        self.num_tiles_y = int(math.ceil(self.rect_height / self.tile_height))

        self.tiles = {}  # Dictionary to store tile labels
        self.tile_positions = {}  # Dictionary to store physical positions in mm
        self.scanning = False
        self.scan_index = 0
        self.runname = None
        self.selected_serial_port = None
        self.selected_camera = None
        self.resume_index = 0  # Store last scanned tile index

        # Signal handler
        self.capture_signal = CaptureSignal()
        self.capture_signal.tile_captured.connect(self.update_tile_color)
        self.robot = None
        self.cap = None
        self.row_images = []
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
        self.run_button = QPushButton("Run")
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
        """Resets the scan and clears the UI."""
        self.stop_action()
        self.scanning = False
        self.scan_index = 0
        self.progress_bar.setValue(0)
        self.runname = None  # Clear run name

        # Reset tile colors to white
        for label in self.tiles.values():
            label.setStyleSheet(
                "border: 1px solid black; background-color: white; padding: 5px;"
            )

        self.log_to_console("Scan reset: Grid cleared and ready for a new run.")

    def run_action(self):
        """Prompt user for a run name and start scanning."""
        if self.robot is None:
            self.log_to_console(
                "Error: No robot selected. Please select a serial port before running."
            )
            return

        runname, ok = QInputDialog.getText(self, "Run Name", "Enter run name:")

        if ok and runname.strip():
            self.runname = runname
            self.log_to_console(f"Run started with name: {runname}")
            self.scan_grid()
        else:
            self.log_to_console("Run canceled or invalid name entered.")

    def resume_action(self):
        """Resume the scan from the last stopped position."""
        if not self.runname:
            self.log_to_console(
                "No previous run found to resume. Start a new run instead."
            )
            return

        self.log_to_console(f"Resuming scan from tile {self.resume_index}...")
        self.scan_index = self.resume_index  # Resume from last index
        self.scan_grid()

    def stop_action(self):
        """Stop scanning process and release the camera."""
        if self.scanning:
            self.timer.stop()
            self.scanning = False
            self.log_to_console("Scanning stopped by user.")

        # Release the camera if it is open
        if self.selected_camera is not None:
            cap = cv2.VideoCapture(self.selected_camera)
            if cap.isOpened():
                cap.release()
                self.log_to_console("Camera released.")

    def scan_grid(self):
        """Iterate through the grid in a snake pattern and capture images."""
        if self.scanning:
            return

        self.scanning = True
        self.scan_index = 0
        total_positions = self.num_tiles_x * self.num_tiles_y
        self.progress_bar.setMaximum(total_positions)
        self.progress_bar.setValue(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_next_tile)
        self.timer.start(500)

    def capture_next_tile(self):
        """Move robot to next tile and capture an image."""
        if self.scan_index >= self.num_tiles_x * self.num_tiles_y:
            self.timer.stop()
            self.scanning = False
            self.log_to_console("Scanning complete.")
            return

        row = self.scan_index // self.num_tiles_x
        col = (
            (self.scan_index % self.num_tiles_x)
            if row % 2 == 0
            else (self.num_tiles_x - 1 - (self.scan_index % self.num_tiles_x))
        )

        x, y = self.tile_positions[(row, col)]
        self.log_to_console(
            f"Moving to ({x:.2f}, {y:.2f}) and capturing image at ({row}, {col})"
        )
        self.robot.go_to(x_position=x, y_position=y, z_position=7.10)

        # Ensure runname is set
        if not self.runname:
            self.log_to_console(
                "Error: Run name is not set. Please provide a run name before scanning."
            )
            return

        base_dir = os.path.join(SystemConstants.SERVER_PATH, self.runname)
        os.makedirs(base_dir, exist_ok=True)
        # Capture an image using the selected camera
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.row_images.append(frame)
            else:
                self.log_to_console(f"Error capturing image at ({row}, {col})")

        self.capture_signal.tile_captured.emit(row, col)
        self.progress_bar.setValue(self.scan_index + 1)
        self.resume_index = self.scan_index + 1  # Store last scanned position
        self.scan_index += 1

        # Check if the row is complete and log it
        if self.scan_index % self.num_tiles_x == 0:
            for col, image in enumerate(self.row_images):
                image_path = os.path.join(base_dir, f"tile_{row}_{col}.jpg")
                cv2.imwrite(image_path, image)
            self.log_to_console(
                f"Row {row} capture complete. Images saved in '{base_dir}/'"
            )
            self.row_images = []

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
        """Update tile color to green when captured."""
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
            self.log_to_console(f"Selected Serial Port: {self.selected_serial_port}")
            self.robot = Robot(port=self.selected_serial_port)
            self.robot.begin()

    def select_camera(self):
        """Opens a dialog to select a camera."""
        cameras = []
        for i in range(2):
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

    def log_to_console(self, message):
        """Logs messages to the embedded console output."""
        self.console_output.append(message)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    top_left = (108.2, 129.6)
    bottom_right = (119.4, 118.8)

    ex = ScanUI(top_left, bottom_right)
    sys.exit(app.exec_())
