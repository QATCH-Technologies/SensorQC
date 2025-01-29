class Position:
    def __init__(self, x: float = None, y: float = None, z: float = None, location_name: str = "DefaultLocation"):
        self._x = x
        self._y = y
        self._z = z
        self._location_name = location_name

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value: float):
        self._validate_coordinate(value, 'x')
        self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value: float):
        self._validate_coordinate(value, 'y')
        self._y = value

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value: float):
        self._validate_coordinate(value, 'z')
        self._z = value

    @property
    def location_name(self):
        return self._location_name

    @location_name.setter
    def location_name(self, value: str):
        if value is not None and not isinstance(value, str):
            raise ValueError("Location name must be a string.")
        self._location_name = value

    def _validate_coordinate(self, value, axis):
        if value is not None and (not isinstance(value, (int, float))):
            raise ValueError(
                f"Invalid value for {axis}: must be an integer or floating point value.")

    def set_position(self, x: float = None, y: float = None, z: float = None, location_name: str = None):
        """Set multiple attributes at once."""
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if z is not None:
            self.z = z
        if location_name is not None:
            self.location_name = location_name

    def get_position(self):
        """Return the position as a tuple (x, y, z, location_name)."""
        return self._x, self._y, self._z, self._location_name

    def reset(self):
        """Reset all attributes to None."""
        self._x = None
        self._y = None
        self._z = None
        self._location_name = None

    def is_valid(self):
        """Check if the position is fully defined (coordinates and location name)."""
        return self._x is not None and self._y is not None and self._z is not None

    def __repr__(self):
        return f"Position(x={self._x}, y={self._y}, z={self._z}, location_name='{self._location_name}')"
