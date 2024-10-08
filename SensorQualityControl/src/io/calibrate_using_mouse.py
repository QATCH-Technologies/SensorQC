import serial
import time
import cv2
import numpy as np
from pynput.mouse import Listener, Controller

# Replace with your G-code machine's COM port and baud rate
ser = serial.Serial()
ser.port = "COM1"
ser.baudrate = 115200  # Set the appropriate baud rate
time.sleep(2)  # Wait for connection to establish

# Initial positions and speed multiplier
current_x = 0
current_y = 0
current_z = 0
mouse_speed_multiplier = 0.05  # Adjust this value for sensitivity

# Mouse control
mouse = Controller()


def send_gcode(command):
    """Send a G-code command to the machine."""
    ser.write((command + "\n").encode())
    response = ser.readline().decode().strip()
    print(response)


def move_x(distance):
    """Move the machine along the X axis."""
    global current_x
    g_code = f"G0 X{distance:.2f}"
    print(g_code)
    send_gcode(g_code)
    current_x += distance


def move_y(distance):
    """Move the machine along the Y axis."""
    global current_y
    g_code = f"G0 Y{distance:.2f}"
    print(g_code)
    send_gcode(g_code)
    current_y += distance


def move_z(distance):
    """Move the machine along the Z axis."""
    global current_z
    g_code = f"G0 Z{distance:.2f}"
    print(g_code)
    send_gcode(g_code)
    current_z += distance


def init_params():
    """Initialize G-code parameters."""
    units_selection = "G21"  # Set units to millimeters
    send_gcode(units_selection)
    positioning_relative = "G91"  # Set to relative positioning
    send_gcode(positioning_relative)


def lock_mouse_to_center(frame):
    """Lock mouse to center of the screen or crosshairs."""
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    mouse.position = (center_x, center_y)
    return center_x, center_y


def on_move(x, y):
    """Handle mouse movement and update X, Y positions."""
    center_x, center_y = mouse.position
    move_x((x - center_x) * mouse_speed_multiplier)
    move_y((center_y - y) * mouse_speed_multiplier)
    # Reset mouse position to the center
    mouse.position = (center_x, center_y)


def on_scroll(x, y, dx, dy):
    """Handle mouse scroll wheel to control Z axis."""
    move_z(dy * mouse_speed_multiplier)


def display_camera_feed():
    """Display the live camera feed with crosshairs in full screen."""
    cap = cv2.VideoCapture(1)  # Change the index if necessary

    # Create a named window for the camera feed
    cv2.namedWindow("Camera Feed", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        height, width, _ = frame.shape
        center_x, center_y = lock_mouse_to_center(frame)

        # Draw crosshairs
        cv2.line(
            frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2
        )  # Horizontal
        cv2.line(
            frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 2
        )  # Vertical

        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    ser.open()
    init_params()

    # Start the camera feed in a separate thread
    from threading import Thread

    camera_thread = Thread(target=display_camera_feed)
    camera_thread.start()

    # Start mouse listener
    with Listener(on_move=on_move, on_scroll=on_scroll) as listener:
        listener.join()

    ser.close()
    camera_thread.join()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program interrupted.")
    finally:
        ser.close()
