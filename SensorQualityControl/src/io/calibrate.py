import keyboard
import serial
import time
import cv2
import numpy as np

STEP = 10
SCALE_FACTOR = 0.1
# Replace with your G-code machine's COM port and baud rate
ser = serial.Serial()
ser.port = "COM4"
ser.baudrate = 115200  # Set the appropriate baud rate
time.sleep(2)  # Wait for connection to establish

# Initial positions
current_x = 0
current_y = 0
current_z = 0


def map_to_machine_axis(coordinate):
    # Map the camera coordinates to G-code workspace
    return coordinate * SCALE_FACTOR


def send_gcode(command):
    """Send a G-code command to the machine."""
    ser.write((command + "\n").encode())
    # time.sleep()  # Wait for the command to be processed
    response = ser.readline().decode().strip()
    print(response)


def move_x(distance):
    """Move the machine along the X axis."""
    global current_x
    g_code = f"G0 X{distance}"
    print(g_code)
    send_gcode(g_code)
    current_x += distance  # Update the current position


def move_y(distance):
    """Move the machine along the Y axis."""
    global current_y
    g_code = f"G0 Y{distance}"
    print(g_code)
    send_gcode(g_code)
    current_y += distance  # Update the current position


def move_z(distance):
    """Move the machine along the Z axis."""
    global current_z
    g_code = f"G0 Z{distance}"
    print(g_code)
    send_gcode(g_code)
    current_z += distance  # Update the current position


def init_params():
    """Initialize G-code parameters."""
    units_selection = "G21"  # Set units to millimeters
    send_gcode(units_selection)

    positioning_relative = "G91"  # Set to relative positioning
    send_gcode(positioning_relative)

    # xy_plane = "G17"  # Select XY plane
    # send_gcode(xy_plane)


def move_to_initial_position():
    """Move the machine to the initial position (100, 100)."""
    move_x(0)  # Move X to 100
    move_y(0)  # Move Y to 100


def log_position():
    """Log the current position of the machine."""
    print(f"Final Position - X: {current_x}, Y: {current_y}, Z: {current_z}")


def display_camera_feed():
    """Display the live camera feed with crosshairs in full screen."""
    cap = cv2.VideoCapture(1)  # Change the index if necessary

    # Create a named window for the camera feed
    cv2.namedWindow("Camera Feed", cv2.WND_PROP_FULLSCREEN)
    # Set the window to fullscreen
    cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Get the dimensions of the frame
        height, width, _ = frame.shape

        # Calculate the center of the frame
        center_x, center_y = width // 2, height // 2

        # Draw the crosshairs
        cv2.line(
            frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2
        )  # Horizontal line
        cv2.line(
            frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 2
        )  # Vertical line

        # Display the resulting frame in full screen
        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    print("Use arrow keys to move the machine.")
    print("Press 'Enter' to log the current position.")
    print("Press 'q' to quit.")
    ser.open()
    init_params()
    move_to_initial_position()  # Move to (100, 100) initially

    # Start the camera feed in a separate thread
    from threading import Thread

    camera_thread = Thread(target=display_camera_feed)
    camera_thread.start()

    while True:
        if keyboard.is_pressed("up"):
            move_y(STEP)  # Adjust this value for speed
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
        elif keyboard.is_pressed("z"):
            move_z(STEP)  # Move up
            time.sleep(0.5)
        elif keyboard.is_pressed("x"):
            move_z(-STEP)  # Move down
            time.sleep(0.5)
        elif keyboard.is_pressed("enter"):
            log_position()  # Log the current position
            time.sleep(0.5)  # Prevent logging multiple times in quick succession
        elif keyboard.is_pressed("q"):
            print("Exiting...")
            break

    ser.close()
    camera_thread.join()  # Wait for the camera thread to finish


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program interrupted.")
    finally:
        ser.close()  # Ensure the serial connection is closed
