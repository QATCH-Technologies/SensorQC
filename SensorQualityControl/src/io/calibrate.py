import keyboard
import serial
import time
import cv2
import numpy as np
from pynput.mouse import Listener  # Use pynput for scroll detection
from datetime import datetime

INITIAL_POSITION = (114.00, 139.10, 59.75, 0.00)
X = INITIAL_POSITION[0]
Y = INITIAL_POSITION[1]
Z = INITIAL_POSITION[2]
FEED_RATE = 800

STEP = 0.1  # Fine movement step
LARGE_STEP = 2.0  # Large movement step
SCALE_FACTOR = 0.1
ZOOM_STEP = 1.0  # Amount to zoom per scroll step
FINE_ZOOM_STEP = 0.1  # Fine zoom step for z key
ser = serial.Serial()
ser.port = "COM4"
ser.baudrate = 115200
time.sleep(2)

current_x = 0
current_y = 0
current_z = 0


def map_to_machine_axis(coordinate):
    return coordinate * SCALE_FACTOR


def get_abs_position():
    ser.write(b"M114\n")
    time.sleep(0.5)
    response = ser.readline().decode("utf-8").strip()
    print(response)
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save the response to a new file with a timestamp
    with open("gcode_position.txt", "a") as file:  # Open in append mode
        file.write(f"{timestamp}: {response}\n")  # Write the timestamp and response


def send_gcode(command):
    ser.write((command + "\n").encode())
    response = ser.readline().decode().strip()
    print(response)


def move_x(distance):
    global current_x
    g_code = f"G0 X{distance}"
    send_gcode(g_code)
    current_x += distance


def move_y(distance):
    global current_y
    g_code = f"G0 Y{distance}"
    send_gcode(g_code)
    current_y += distance


def move_z(distance):
    global current_z
    g_code = f"G0 Z{distance}"
    send_gcode(g_code)
    current_z += distance


def init_params():
    send_gcode("G21")
    send_gcode("G91")


def display_camera_feed():
    cap = cv2.VideoCapture(1)
    cv2.namedWindow("Camera Feed", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2
        cv2.line(
            frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2
        )
        cv2.line(
            frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 2
        )

        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def on_scroll(x, y, dx, dy):
    """Handle scroll wheel zoom (Z-axis movement)."""
    if dy > 0:  # Scroll up
        move_z(ZOOM_STEP)
    elif dy < 0:  # Scroll down
        move_z(-ZOOM_STEP)


def main():
    print("Use WASD keys for large movements (W: up, A: left, S: down, D: right).")
    print("Use arrow keys for fine movements (↑: up, ↓: down, ←: left, →: right).")
    print("Scroll to zoom (Z-axis).")
    print("Press 'z' to zoom in (fine).")
    print("Press 'x' to zoom out (fine).")
    print("Press 'Enter' to log the current position.")
    print("Press 'q' to quit.")
    ser.open()
    init_params()

    from threading import Thread

    camera_thread = Thread(target=display_camera_feed)
    camera_thread.start()

    # Start listening for scroll events using pynput
    scroll_listener = Listener(on_scroll=on_scroll)
    scroll_listener.start()

    while True:
        if keyboard.is_pressed("w"):
            move_y(LARGE_STEP)
            time.sleep(0.5)
        elif keyboard.is_pressed("s"):
            move_y(-LARGE_STEP)
            time.sleep(0.5)
        elif keyboard.is_pressed("a"):
            move_x(-LARGE_STEP)
            time.sleep(0.5)
        elif keyboard.is_pressed("d"):
            move_x(LARGE_STEP)
            time.sleep(0.5)
        elif keyboard.is_pressed("up"):
            move_y(STEP)
            time.sleep(0.5)
        elif keyboard.is_pressed("down"):
            move_y(-STEP)
            time.sleep(0.5)
        elif keyboard.is_pressed("left"):
            move_x(-STEP)
            time.sleep(0.5)
        elif keyboard.is_pressed("right"):
            move_x(STEP)
            time.sleep(0.5)
        elif keyboard.is_pressed("z"):  # Fine zoom in
            move_z(FINE_ZOOM_STEP)
            time.sleep(0.5)
        elif keyboard.is_pressed("x"):  # Fine zoom out
            move_z(-FINE_ZOOM_STEP)
            time.sleep(0.5)
        elif keyboard.is_pressed("enter"):
            get_abs_position()
            time.sleep(0.5)
        elif keyboard.is_pressed("q"):
            print("Exiting...")
            break

    ser.close()
    camera_thread.join()
    scroll_listener.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program interrupted.")
    finally:
        ser.close()
