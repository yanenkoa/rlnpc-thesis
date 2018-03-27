from abc import ABC, abstractmethod
from enum import Enum
import enum
from math import sin, cos, sqrt, pi
from typing import List, NamedTuple

from Util import Vector2, RectangleAABB, get_rectangle_points, make_rectangle, rectangles_intersect, \
    point_in_rectangle_aabb, RectangleCollision, Ray, Collision, ray_rectangle_aabb_intersect, LineSegment, \
    segment_aabb_intersect


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

    _reward_loss_ps: float = 0.1
    _reward_lost_heat_coef: float = 1

    _init_angle: float
    _init_position: Vector2
    _angle: float
    _position: Vector2
    _move_speed_ps: float
    _gold: int
    _heat: float
    _reward: float

    def __init__(self, init_pos: Vector2, init_angle: float, move_speed_ps: float):
        self._init_angle = self._angle = init_angle
        self._init_position = init_pos
        self._position = Vector2(init_pos.x, init_pos.y)
        self._move_speed_ps = move_speed_ps
        self._gold = 0
        self._heat = 0
        self._reward = 0

    def get_translation(self, elapsed_time_s: float, direction: PlayerMovementDirection) -> Vector2:
        dx = cos(self._angle) * self._move_speed_ps * elapsed_time_s * direction.value
        dy = sin(self._angle) * self._move_speed_ps * elapsed_time_s * direction.value
        return Vector2(dx, dy)

    def apply_translation(self, translation: Vector2) -> None:
        self._position += translation

    def update_angle(self, new_angle: float) -> None:
        self._angle = new_angle

    def add_gold(self, gold) -> None:
        self._reward += gold
        self._gold += gold

    def get_rectangle(self) -> RectangleAABB:
        return make_rectangle(self._position, self.width, self.height)

    def set_heat(self, heat: float) -> None:
        self._heat = heat

    def update_reward(self, elapsed_time_s: float) -> None:
        self._reward -= elapsed_time_s * (self._reward_loss_ps + self._heat * self._reward_lost_heat_coef)

    def reset_after_step(self) -> None:
        self._reward = 0

    def reset(self) -> None:
        self._angle = self._init_angle
        self._position = Vector2(self._init_position.x, self._init_position.y)
        self._gold = 0
        self._heat = 0
        self._reward = 0

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

    @property
    def reward(self):
        return self._reward


class WallType(Enum):
    RECTANGLE = 0


class Wall(ABC):

    @abstractmethod
    def point_collides(self, point: Vector2) -> bool: pass

    @abstractmethod
    def ray_collides(self, ray: Ray) -> Collision: pass

    @abstractmethod
    def segment_collides(self, line_segment: LineSegment) -> Collision: pass

    @abstractmethod
    def get_type(self) -> WallType: pass


class WallColliderValidator(LocationValidator):
    _walls: List[Wall]

    def __init__(self, walls: List[Wall]):
        self._walls = walls

    def is_valid(self, point: Vector2) -> bool:
        for wall in self._walls:
            if wall.point_collides(point):
                return False
        return True


class RectangleWall(Wall):
    _rectangle: RectangleAABB

    def __init__(self, rectangle: RectangleAABB):
        self._rectangle = rectangle

    def point_collides(self, point: Vector2) -> bool:
        return point_in_rectangle_aabb(point, self._rectangle)

    def ray_collides(self, ray: Ray) -> Collision:
        return ray_rectangle_aabb_intersect(ray, self._rectangle)

    def segment_collides(self, line_segment: LineSegment) -> Collision:
        return segment_aabb_intersect(line_segment, self._rectangle)

    def get_type(self) -> WallType:
        return WallType.RECTANGLE

    @property
    def rectangle(self):
        return self._rectangle

    def __str__(self):
        return str(self._rectangle)


class GoldChest:
    width: float = 10
    height: float = 10

    _collected: bool = False
    _gold: int
    _location: Vector2
    _rectangle: RectangleAABB

    def __init__(self, gold: int, location: Vector2):
        self._gold = gold
        self._location = location
        self._rectangle = make_rectangle(location, self.width, self.height)

    def collect(self) -> int:
        self._collected = True
        return self._gold

    def reset(self) -> None:
        self._collected = False

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
    _walls: List[Wall]

    _radius: float = 5

    def __init__(self, heat: float, location: Vector2, walls: List[Wall]):
        self._heat = heat
        self._location = location
        self._walls = walls

    def get_heat(self, other_location: Vector2):

        segment_to = LineSegment(self._location, other_location)
        for wall in self._walls:
            if wall.segment_collides(segment_to).intersects:
                return 0

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
    _rect: RectangleAABB

    def __init__(self, location: Vector2):
        self._location = location
        self._rect = make_rectangle(self._location, self.width, self.height)

    @property
    def location(self):
        return self._location

    @property
    def rectangle(self):
        return self._rect


class SensedObject(Enum):
    NONE = enum.auto()
    WALL = enum.auto()
    GOLD = enum.auto()
    PORTAL = enum.auto()


class ProximitySensor:
    _player: Player
    _angle: float
    _max_distance: float
    _walls: List[Wall]
    _gold_chests: List[GoldChest]
    _portal: Portal

    _point: Vector2
    _current_obj: SensedObject
    _segment: LineSegment

    def __init__(self,
                 player: Player,
                 angle: float,
                 max_distance: float,
                 walls: List[Wall],
                 gold_chests: List[GoldChest],
                 portal: Portal):
        self._player = player
        self._angle = angle
        self._max_distance = max_distance
        self._walls = walls
        self._gold_chests = gold_chests
        self._portal = portal

        segment_end = self._get_segment_end()
        self._point = segment_end
        self._current_obj = SensedObject.NONE
        self._segment = LineSegment(self._player.position, segment_end)

    def update_point(self) -> None:

        segment_end = self._get_segment_end()
        self._segment = LineSegment(self._player.position, segment_end)

        min_collision = Collision(False, None)
        min_distance = float("inf")
        for wall in self._walls:
            current_collision = wall.segment_collides(self._segment)
            if current_collision.intersects:
                distance = abs(current_collision.point - self._segment.a)
                if distance < min_distance:
                    min_collision = current_collision
                    min_distance = distance
                    self._current_obj = SensedObject.WALL

        for chest in self._gold_chests:
            if chest.collected:
                continue
            current_collision = segment_aabb_intersect(self._segment, chest.rectangle)
            if current_collision.intersects:
                distance = abs(current_collision.point - self._segment.a)
                if distance < min_distance:
                    min_collision = current_collision
                    min_distance = distance
                    self._current_obj = SensedObject.GOLD

        current_collision = segment_aabb_intersect(self._segment, self._portal.rectangle)
        if current_collision.intersects:
            distance = abs(current_collision.point - self._segment.a)
            if distance < min_distance:
                min_collision = current_collision
                min_distance = distance
                self._current_obj = SensedObject.PORTAL

        if min_distance <= self._max_distance:
            self._point = min_collision.point
        else:
            self._point = segment_end
            self._current_obj = SensedObject.NONE

    def _get_segment_end(self) -> Vector2:
        return self._player.position + Vector2(cos(self._angle), sin(self._angle)) * self._max_distance

    def reset(self) -> None:
        segment_end = self._get_segment_end()
        self._point = segment_end
        self._current_obj = SensedObject.NONE
        self._segment = LineSegment(self._player.position, segment_end)

    @property
    def point(self):
        return self._point

    @property
    def sensed_obj(self):
        return self._current_obj


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
    _proximity_sensors: List[ProximitySensor]

    def __init__(
            self,
            width: float,
            height: float,
            player: Player,
            walls: List[Wall],
            gold_chests: List[GoldChest],
            heat_sources: List[HeatSource],
            portal: Portal,
            proximity_sensors: List[ProximitySensor]):
        self._width = width
        self._height = height
        self._player = player
        self._walls = walls
        self._gold_chests = gold_chests
        self._heat_sources = heat_sources
        self._portal = portal

        left_wall = RectangleWall(RectangleAABB(Vector2(-100, 0), Vector2(0, height)))
        top_wall = RectangleWall(RectangleAABB(Vector2(0, height), Vector2(width, height + 100)))
        right_wall = RectangleWall(RectangleAABB(Vector2(width, 0), Vector2(width + 100, height)))
        bottom_wall = RectangleWall(RectangleAABB(Vector2(0, -100), Vector2(width, 0)))

        self._walls.extend([left_wall, top_wall, right_wall, bottom_wall])

        self._validator = ValidatorComposition(
            [
                WallColliderValidator(self._walls),
                RectangleConstraints(self._width, self._height),
            ]
        )
        self._game_over = False

        self._proximity_sensors = proximity_sensors

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
                translation = Vector2(0, 0)
        self._player.apply_translation(translation)

    def update_state(self, elapsed_time_s: float) -> None:

        portal_rect = self._portal.rectangle
        player_rect = self._player.get_rectangle()
        if rectangles_intersect(player_rect, portal_rect).intersects:
            self._game_over = True

        if self._game_over:
            return

        for gold_chest in self._gold_chests:
            if not gold_chest.collected and rectangles_intersect(gold_chest.rectangle, player_rect).intersects:
                self._player.add_gold(gold_chest.collect())

        player_pos = self._player.position
        heat = sum(hs.get_heat(player_pos) for hs in self._heat_sources)
        self._player.set_heat(heat)

        for prox_sens in self._proximity_sensors:
            prox_sens.update_point()

        self._player.update_reward(elapsed_time_s)

    def reset(self) -> None:
        self._game_over = False
        self._player.reset()
        for chest in self._gold_chests:
            chest.reset()
        for sensor in self._proximity_sensors:
            sensor.reset()

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

    @property
    def proximity_sensors(self):
        return self._proximity_sensors
