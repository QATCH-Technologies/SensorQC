import threading
import mouse
import keyboard
from robot import Robot
from dino_lite_edge import Camera, Microscope

Z_STEP = 0.1
Z_SPEED = 1.0
SCROLL_SENSITIVITY = 10.0
X_STEP = 0.5
Y_STEP = 0.5
BRIGHT_FIELD = 1
DARK_FIELD = 2


class Controls:
    def __init__(self, robot: Robot, camera: Camera, microscope: Microscope):
        self.__robot__ = robot
        self.__camera__ = camera
        self.__microscope__ = microscope
        self.__robot__.begin()
        self.__robot__.relative_mode()
        self.__axis_state__ = {"X": False, "Y": False, "Z": False}

    def move_z(self, direction, speed):
        z_movement = direction * speed * 1.0
        return self.__robot__.translate_z(z_movement)

    def zoom_in(self, distance):
        while self.__axis_state__["Z"]:
            self.__robot__.translate_z(distance)

    def zoom_out(self, distance):
        while self.__axis_state__["Z"]:
            self.__robot__.translate_z(-distance)

    def move_up(self, distance):
        while self.__axis_state__["Y"]:
            self.__robot__.translate_y(distance)

    def move_down(self, distance):
        while self.__axis_state__["Y"]:
            self.__robot__.translate_y(-distance)

    def move_left(self, distance):
        while self.__axis_state__["X"]:
            self.__robot__.translate_x(-distance)

    def move_right(self, distance):
        while self.__axis_state__["X"]:
            self.__robot__.translate_x(distance)

    def key_press(self, key):
        print(f"Exposure: {scope.get_exposure()}")
        if key == "w" and not self.__axis_state__["Y"]:
            self.__axis_state__["Y"] = True
            threading.Thread(target=self.move_up, args=(Y_STEP,), daemon=True).start()
        elif key == "s" and not self.__axis_state__["Y"]:
            self.__axis_state__["Y"] = True
            threading.Thread(target=self.move_down, args=(Y_STEP,), daemon=True).start()
        elif key == "a" and not self.__axis_state__["X"]:
            self.__axis_state__["X"] = True
            threading.Thread(target=self.move_left, args=(X_STEP,), daemon=True).start()
        elif key == "d" and not self.__axis_state__["X"]:
            self.__axis_state__["X"] = True
            threading.Thread(
                target=self.move_right, args=(X_STEP,), daemon=True
            ).start()
        elif key == "q" and not self.__axis_state__["Z"]:
            self.__axis_state__["Z"] = True
            threading.Thread(target=self.zoom_in, args=(Z_STEP,), daemon=True).start()
        elif key == "e" and not self.__axis_state__["Z"]:
            self.__axis_state__["Z"] = True
            threading.Thread(target=self.zoom_out, args=(Z_STEP,), daemon=True).start()

    def key_release(self, key):
        if key == "w" or key == "s":
            self.__axis_state__["Y"] = False
        elif key == "a" or key == "d":
            self.__axis_state__["X"] = False
        elif key == "q" or key == "e":
            self.__axis_state__["Z"] = False

    def on_scroll(self, event):
        # Scroll direction: positive for up, negative for down
        direction = 1 if event.delta > 0 else -1
        # Absolute scroll value (speed) will determine movement size
        # Adjust divisor to tune sensitivity
        speed = abs(event.delta) / SCROLL_SENSITIVITY
        self.move_z(direction, speed)

    def control(self):
        print(
            "Use WASD keys to move X/Y, mouse scroll to move Z, and Enter to get absolute position. Press Esc to quit."
        )

        # mouse.hook(
        #     lambda e: self.on_scroll(e) if isinstance(e, mouse.WheelEvent) else None
        # )

        # Set up key press and release events
        keyboard.on_press_key("a", lambda e: self.key_press("a"))
        keyboard.on_press_key("s", lambda e: self.key_press("s"))
        keyboard.on_press_key("d", lambda e: self.key_press("d"))
        keyboard.on_press_key("w", lambda e: self.key_press("w"))
        keyboard.on_press_key("q", lambda e: self.key_press("q"))
        keyboard.on_press_key("e", lambda e: self.key_press("e"))

        keyboard.on_release_key("w", lambda e: self.key_release("w"))
        keyboard.on_release_key("a", lambda e: self.key_release("a"))
        keyboard.on_release_key("s", lambda e: self.key_release("s"))
        keyboard.on_release_key("d", lambda e: self.key_release("d"))
        keyboard.on_release_key("q", lambda e: self.key_release("q"))
        keyboard.on_release_key("e", lambda e: self.key_release("e"))
        keyboard.on_press_key("enter", lambda e: self.__robot__.get_absolute_position())
        keyboard.on_press_key("z", lambda e: self.__camera__.toggle_camera())

        keyboard.on_press_key("r", lambda e: self.__microscope__.led_off())
        keyboard.on_press_key(
            "t", lambda e: self.__microscope__.led_on(state=BRIGHT_FIELD)
        )
        keyboard.on_press_key(
            "y", lambda e: self.__microscope__.led_on(state=DARK_FIELD)
        )
        keyboard.wait("esc")
        self.__robot__.end()
        self.__microscope__.end()


if __name__ == "__main__":
    scope = Microscope()
    scope.set_autoexposure(state=0)

    # print(scope.get_autoexposure())

    # print(scope.get_fov_index())
    # print(scope.get_config())
    scope.set_exposure(828)
    # print(scope.get_exposure())
    cam = Camera()
    rob = Robot(port="COM4", debug=False)
    controls = Controls(rob, cam, scope)
    controls.control()
