from abc import ABC, abstractmethod
from enum import Enum
from math import sin, cos, pi
from time import time
from typing import List

from InputDevice import InputDevice
from Util import Vector2, Rectangle, make_rectangle, get_rectangle_points, point_in_rectangle_points


class GameObject(ABC):

    @abstractmethod
    def update(self) -> None: pass


class LocationValidator(ABC):

    @abstractmethod
    def is_valid(self, position: Vector2) -> bool: pass


class ValidatorComposition(LocationValidator):

    _validators: List[LocationValidator]

    def __init__(self, validators: List[LocationValidator]):
        self._validators = validators

    def is_valid(self, position: Vector2):
        for validator in self._validators:
            if not validator.is_valid(position):
                return False
        return True


class RectangleConstraints(LocationValidator):

    _x_cap: float
    _y_cap: float

    def __init__(self, x_cap: float, y_cap: float):
        self._x_cap = x_cap
        self._y_cap = y_cap

    def is_valid(self, position: Vector2) -> bool:
        return 0 <= position.x <= self._x_cap and 0 <= position.y <= self._y_cap


class Player(GameObject):

    width: float = 20
    height: float = 20

    _angle: float
    _position: Vector2
    _move_speed_ps: float
    _angle_speed_ps: float
    _input_device: InputDevice
    _last_update_s: float

    validator: LocationValidator

    def __init__(
            self,
            init_pos: Vector2,
            move_speed_ps: float,
            angle_speed_ps: float,
            input_device: InputDevice,
            validator: LocationValidator = None
    ):
        self._angle = 0
        self._position = init_pos
        self._move_speed_ps = move_speed_ps
        self._angle_speed_ps = angle_speed_ps
        self._input_device = input_device
        self._last_update_s = time()
        self.validator = validator

    def _get_move_increment(self, elapsed_time_s: float) -> Vector2:
        dx = cos(self._angle) * self._move_speed_ps * elapsed_time_s
        dy = sin(self._angle) * self._move_speed_ps * elapsed_time_s
        return Vector2(dx, dy)

    def _move_forward(self, elapsed_time_s: float) -> None:
        move_increment = self._get_move_increment(elapsed_time_s)
        self._position = Vector2(
            self._position.x + move_increment.x,
            self._position.y + move_increment.y
        )

    def _move_backward(self, elapsed_time_s: float) -> None:
        move_increment = self._get_move_increment(elapsed_time_s)
        self._position = Vector2(
            self._position.x - move_increment.x,
            self._position.y - move_increment.y
        )

    def _turn_counter_clockwise(self, elapsed_time_s) -> None:
        self._angle += self._angle_speed_ps * elapsed_time_s

    def _turn_clockwise(self, elapsed_time_s) -> None:
        self._angle -= self._angle_speed_ps * elapsed_time_s

    def update(self) -> None:

        old_position = Vector2(self._position.x, self._position.y)

        cur_time_s = time()
        elapsed_time_s = cur_time_s - self._last_update_s
        self._last_update_s = cur_time_s

        if self._input_device.is_key_down("w"):
            self._move_forward(elapsed_time_s)

        if self._input_device.is_key_down("s"):
            self._move_backward(elapsed_time_s)

        if self._input_device.is_key_down("a"):
            self._turn_counter_clockwise(elapsed_time_s)

        if self._input_device.is_key_down("d"):
            self._turn_clockwise(elapsed_time_s)

        if self.validator is not None:
            player_points = get_rectangle_points(make_rectangle(self._position, self.width, self.height))
            for p in player_points:
                if not self.validator.is_valid(p):
                    self._position = old_position
                    return


    @property
    def angle(self) -> float:
        return self._angle

    @property
    def position(self) -> Vector2:
        return self._position


class WallType(Enum):
    RECTANGLE = 0


class Wall(ABC):

    @abstractmethod
    def point_collides(self, point: Vector2) -> bool: pass

    @abstractmethod
    def get_type(self) -> WallType: pass


class WallColliderValidator(LocationValidator):

    _walls: List[Wall]

    def __init__(self, walls: List[Wall]):
        self._walls = walls

    def is_valid(self, point: Vector2):
        for wall in self._walls:
            if wall.point_collides(point):
                return False
        return True


class RectangleWall(Wall):

    _rectangle: Rectangle

    def __init__(self, rectangle: Rectangle):
        self._rectangle = rectangle

    def point_collides(self, point: Vector2) -> bool:
        return point_in_rectangle_points(point, get_rectangle_points(self._rectangle))

    def get_type(self) -> WallType:
        return WallType.RECTANGLE

    @property
    def rectangle(self):
        return self._rectangle


class World:

    _width: int
    _height: int
    _player: Player
    _walls: List[Wall]

    def __init__(self, width: float, height: float, player: Player, walls: List[Wall]):
        self._width = width
        self._height = height
        self._walls = walls
        self._player = player
        self._player.validator = ValidatorComposition(
            [
                WallColliderValidator(self._walls),
                RectangleConstraints(self._width, self._height)
            ]
        )

    def update(self) -> None:
        self._player.update()

    @property
    def player(self) -> Player:
        return self._player

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def walls(self) -> List[Wall]:
        return self._walls
