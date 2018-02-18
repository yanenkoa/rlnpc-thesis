from abc import ABC, abstractmethod
from enum import Enum
from math import sin, cos
from typing import List, NamedTuple

from Util import Vector2, Rectangle, get_rectangle_points, point_in_rectangle_points, make_rectangle, \
    rectangles_intersect


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


class PlayerUpdate(NamedTuple):
    translation: Vector2
    angle_increment: float


class PlayerMovementDirection(Enum):
    BACKWARD: int = -1
    NONE: int = 0
    FORWARD: int = 1


class Player:
    width: float = 15.0
    height: float = 15.0

    _angle: float
    _position: Vector2
    _move_speed_ps: float
    _gold: int
    _heat: float

    def __init__(self, init_pos: Vector2, init_angle: float, move_speed_ps: float):
        self._position = init_pos
        self._angle = init_angle
        self._move_speed_ps = move_speed_ps
        self._gold = 0
        self._heat = 0

    def get_translation(self, elapsed_time_s: float, direction: PlayerMovementDirection) -> Vector2:
        dx = cos(self._angle) * self._move_speed_ps * elapsed_time_s * direction.value
        dy = sin(self._angle) * self._move_speed_ps * elapsed_time_s * direction.value
        return Vector2(dx, dy)

    def apply_translation(self, translation: Vector2):
        self._position += translation

    def update_angle(self, new_angle: float) -> None:
        self._angle = new_angle

    def add_gold(self, gold) -> None:
        self._gold += gold

    def get_rectangle(self):
        return make_rectangle(self._position, self.width, self.height)

    def set_heat(self, heat: float):
        self._heat = heat

    @property
    def angle(self) -> float:
        return self._angle

    @property
    def position(self) -> Vector2:
        return self._position

    @property
    def gold(self) -> int:
        return self._gold

    @property
    def heat(self):
        return self._heat


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


class GoldChest:
    width: float = 10
    height: float = 10

    _collected: bool = False
    _gold: int
    _location: Vector2
    _rectangle: Rectangle

    def __init__(self, gold: int, location: Vector2):
        self._gold = gold
        self._location = location
        self._rectangle = make_rectangle(location, self.width, self.height)

    def collect(self) -> int:
        self._collected = True
        return self._gold

    @property
    def location(self):
        return self._location

    @property
    def gold(self):
        return self._gold

    @property
    def collected(self):
        return self._collected

    @property
    def rectangle(self):
        return self._rectangle


class HeatSource:
    _coef: float = 300

    _heat: float
    _location: Vector2
    _radius: float = 5

    def __init__(self, heat: float, location: Vector2):
        self._heat = heat
        self._location = location

    def get_heat(self, other_location: Vector2):
        distance2 = (other_location.x - self._location.x) ** 2 + (other_location.y - self._location.y) ** 2
        if distance2 <= self._radius ** 2:
            return self._heat
        else:
            return self._coef * self._heat / distance2

    @property
    def location(self):
        return self._location

    @property
    def radius(self):
        return self._radius


class Portal:
    width: float = 30
    height: float = 30

    _location: Vector2

    def __init__(self, location: Vector2):
        self._location = location

    @property
    def location(self):
        return self._location


class World:
    _width: int
    _height: int
    _player: Player
    _walls: List[Wall]
    _gold_chests: List[GoldChest]
    _heat_sources: List[HeatSource]
    _portal: Portal
    _validator: LocationValidator
    _game_over: bool

    def __init__(
            self,
            width: float,
            height: float,
            player: Player,
            walls: List[Wall],
            gold_chests: List[GoldChest],
            heat_sources: List[HeatSource],
            portal: Portal):
        self._width = width
        self._height = height
        self._player = player
        self._walls = walls
        self._gold_chests = gold_chests
        self._heat_sources = heat_sources
        self._portal = portal
        self._validator = ValidatorComposition(
            [
                WallColliderValidator(self._walls),
                RectangleConstraints(self._width, self._height),
            ]
        )
        self._game_over = False

    def update_player_angle(self, new_angle: float) -> None:
        if not self._game_over:
            self._player.update_angle(new_angle)

    def move_player(self, elapsed_time_s: float, direction: PlayerMovementDirection) -> None:

        if self._game_over:
            return

        translation = self._player.get_translation(elapsed_time_s, direction)
        new_player_points = get_rectangle_points(
            make_rectangle(
                self._player.position + translation,
                self._player.width,
                self._player.height
            )
        )
        for p in new_player_points:
            if not self._validator.is_valid(p):
                break
        else:
            self._player.apply_translation(translation)

    def update_state(self) -> None:

        portal_rect = make_rectangle(self._portal.location, self._portal.width, self._portal.height)
        player_rect = self._player.get_rectangle()
        if rectangles_intersect(player_rect, portal_rect):
            self._game_over = True

        if self._game_over:
            return

        for gold_chest in self._gold_chests:
            if not gold_chest.collected and rectangles_intersect(gold_chest.rectangle, player_rect):
                self._player.add_gold(gold_chest.collect())

        player_pos = self._player.position
        heat = sum(hs.get_heat(player_pos) for hs in self._heat_sources)
        self._player.set_heat(heat)

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

    @property
    def gold_chests(self) -> List[GoldChest]:
        return self._gold_chests

    @property
    def heat_sources(self) -> List[HeatSource]:
        return self._heat_sources

    @property
    def portal(self) -> Portal:
        return self._portal

    @property
    def game_over(self) -> bool:
        return self._game_over
