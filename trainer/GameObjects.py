from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum
from math import sin, cos
from typing import List, NamedTuple, Tuple, Optional, Set

import numpy as np

from trainer.Util import Vector2, RectangleAABB, get_rectangle_points, make_rectangle, rectangles_intersect, \
    point_in_rectangle_aabb, Ray, Collision, ray_rectangle_aabb_intersect, LineSegment, \
    segment_aabb_intersect, LineSegments, np_line_segments_intersect


class LocationValidator(ABC):

    @abstractmethod
    def is_valid(self, position: Vector2) -> bool: pass


class ValidatorComposition(LocationValidator):
    _validators = ...  # type: List[LocationValidator]

    def __init__(self, validators: List[LocationValidator]):
        self._validators = validators

    def is_valid(self, position: Vector2):
        for validator in self._validators:
            if not validator.is_valid(position):
                return False
        return True


class RectangleConstraints(LocationValidator):
    _x_cap = ...  # type: float
    _y_cap = ...  # type: float

    def __init__(self, x_cap: float, y_cap: float):
        self._x_cap = x_cap
        self._y_cap = y_cap

    def is_valid(self, position: Vector2) -> bool:
        return 0 <= position.x <= self._x_cap and 0 <= position.y <= self._y_cap


PlayerUpdate = namedtuple("PlayerUpdate", [
    "translation",
    "angle_increment"
])
# class PlayerUpdate(NamedTuple):
#     translation: Vector2
#     angle_increment: float


class PlayerMovementDirection(Enum):
    BACKWARD = -1  # type: int
    NONE = 0  # type: int
    FORWARD = 1  # type: int


class Player:
    width = 15.0  # type: float
    height = 15.0  # type: float

    _reward_lost_heat_coef_ps = 1.  # type: float
    _portal_coef = 2.  # type: float
    _portal_reward_collected = False  # type: bool

    _init_angle = ...  # type: float
    _init_position = ...  # type: Vector2
    _angle = ...  # type: float
    _position = ...  # type: Vector2
    _move_speed_ps = ...  # type: float
    _gold = ...  # type: float
    _heat = ...  # type: float
    _reward = ...  # type: float
    _reward_sum = ...  # type: float

    def __init__(self, init_pos: Vector2, init_angle: float, move_speed_ps: float):
        self._init_angle = self._angle = init_angle
        self._init_position = init_pos
        self._position = Vector2(init_pos.x, init_pos.y)
        self._move_speed_ps = move_speed_ps
        self._gold = 0.
        self._heat = 0.
        self._reward = 0.
        self._reward_sum = 0.

    def get_translation(self, elapsed_time_s: float, direction: PlayerMovementDirection) -> Vector2:
        dx = cos(self._angle) * self._move_speed_ps * elapsed_time_s * direction.value
        dy = sin(self._angle) * self._move_speed_ps * elapsed_time_s * direction.value
        return Vector2(dx, dy)

    def apply_translation(self, translation: Vector2) -> None:
        self._position += translation

    def update_angle(self, new_angle: float) -> None:
        self._angle = new_angle

    def add_gold(self, gold: float) -> None:
        self._reward += gold
        self._gold += gold

    def add_portal_reward(self) -> None:
        if not self._portal_reward_collected:
            self._portal_reward_collected = True
            self._reward += self._portal_coef * self._gold

    def get_rectangle(self) -> RectangleAABB:
        return make_rectangle(self._position, self.width, self.height)

    def set_heat(self, heat: float) -> None:
        self._heat = heat

    def apply_passive_reward_penalty(self, elapsed_time_s: float) -> None:
        self._reward -= elapsed_time_s * self._heat * self._reward_lost_heat_coef_ps

    def add_custom_reward(self, custom_reward: float) -> None:
        self._reward += custom_reward

    def reset_reward_after_step(self) -> None:
        self._reward_sum += self._reward
        self._reward = 0

    def reset(self) -> None:
        self._angle = self._init_angle
        self._position = Vector2(self._init_position.x, self._init_position.y)
        self._portal_reward_collected = False
        self._gold = 0
        self._heat = 0.
        self._reward = 0.
        self._reward_sum = 0.

    @property
    def angle(self) -> float:
        return self._angle

    @property
    def position(self) -> Vector2:
        return self._position.clone()

    @property
    def gold(self) -> float:
        return self._gold

    @property
    def heat(self) -> float:
        return self._heat

    @property
    def reward_sum(self):
        return self._reward_sum

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
    _walls = ...  # type: List[Wall]

    def __init__(self, walls: List[Wall]):
        self._walls = walls

    def is_valid(self, point: Vector2) -> bool:
        for wall in self._walls:
            if wall.point_collides(point):
                return False
        return True


class RectangleWall(Wall):
    _rectangle = ...  # type: RectangleAABB

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
    width = 20  # type: float
    height = 20  # type: float

    _collected = False  # type: bool

    _gold = ...  # type: float
    _location = ...  # type: Vector2
    _rectangle = ...  # type: RectangleAABB

    def __init__(self, gold: float, location: Vector2):
        self._gold = gold
        self._location = location
        self._rectangle = make_rectangle(location, self.width, self.height)

    def collect(self) -> float:
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
    _coef = 10  # type: float

    _heat = ...  # type: float
    _location = ...  # type: Vector2
    _walls = ...  # type: List[Wall]

    _radius = 5.  # type: float

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
            return self._coef * self._heat
        else:
            return self._coef * self._heat / np.sqrt(distance2)

    @property
    def location(self):
        return self._location

    @property
    def radius(self):
        return self._radius


class Portal:
    width = 30.
    height = 30.

    _location = ...  # type: Vector2
    _rect = ...  # type: RectangleAABB

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
    NONE = 1
    WALL = 2
    GOLD = 3
    PORTAL = 4


def get_rect_segments_np(rect: RectangleAABB) -> Tuple[np.ndarray, np.ndarray]:
    points = get_rectangle_points(rect)
    first_points = np.array([
        [points.lower_left.x, points.lower_left.y],
        [points.lower_right.x, points.lower_right.y],
        [points.upper_right.x, points.upper_right.y],
        [points.upper_left.x, points.upper_left.y],
    ])
    second_points = np.array([
        [points.lower_right.x, points.lower_right.y],
        [points.upper_right.x, points.upper_right.y],
        [points.upper_left.x, points.upper_left.y],
        [points.lower_left.x, points.lower_left.y],
    ])
    return first_points, second_points


def loop_array(arr: np.ndarray, pre_loop_size: int) -> None:
    size = arr.shape[0]
    indices = np.arange(size) % pre_loop_size
    arr[:] = arr[indices, :]


class ProximitySensors:
    _player = ...  # type: Player
    _angles = ...  # type: np.ndarray
    _max_distance = ...  # type: float
    _walls = ...  # type: List[RectangleWall]
    _gold_chests = ...  # type: List[GoldChest]
    _portal = ...  # type: Portal

    _n_sensors = ...  # type: int
    _end_biases_units = ...  # type: np.ndarray
    _player_loc_np = ...  # type: np.ndarray
    _points_np = ...  # type: np.ndarray
    _distances = ...  # type: np.ndarray
    _current_objects = ...  # type: np.ndarray
    _wall_segments = ...  # type: LineSegments
    _chest_segments = ...  # type: LineSegments
    _portal_segments = ...  # type: LineSegments
    _sensor_segments = ...  # type: LineSegments

    def __init__(self,
                 player: Player,
                 angles: np.ndarray,
                 max_distance: float,
                 walls: List[RectangleWall],
                 gold_chests: List[GoldChest],
                 portal: Portal):
        self._player = player
        self._angles = angles
        self._max_distance = max_distance
        self._walls = walls
        self._gold_chests = gold_chests
        self._portal = portal

        self._n_sensors = angles.size
        player_loc = self._player.position
        player_angle = self._player.angle
        self._end_biases_units = np.vstack((np.cos(self._angles + player_angle),
                                            np.sin(self._angles + player_angle))).T
        self._player_loc_np = np.array([player_loc.x, player_loc.y], dtype=np.float32)
        self._points_np = np.zeros(shape=(self._n_sensors, 2), dtype=np.float32)
        self._distances = np.zeros(shape=(self._n_sensors,), dtype=np.float32)
        self._current_objects = np.zeros(shape=(self._n_sensors,), dtype=np.int32)
        self._current_objects[:] = SensedObject.NONE.value
        self._wall_segments = LineSegments(
            np.empty(shape=(self._n_sensors * len(self._walls) * 4, 2), dtype=np.float32),
            np.empty(shape=(self._n_sensors * len(self._walls) * 4, 2), dtype=np.float32),
        )
        self._chest_segments = LineSegments(
            np.empty(shape=(self._n_sensors * len(self._gold_chests) * 4, 2), dtype=np.float32),
            np.empty(shape=(self._n_sensors * len(self._gold_chests) * 4, 2), dtype=np.float32),
        )
        self._portal_segments = LineSegments(
            np.empty(shape=(self._n_sensors * 4, 2), dtype=np.float32),
            np.empty(shape=(self._n_sensors * 4, 2), dtype=np.float32),
        )
        self._sensor_segments = LineSegments(
            np.empty(shape=(self._n_sensors, 2), dtype=np.float32),
            np.empty(shape=(self._n_sensors, 2), dtype=np.float32),
        )
        self._init_segments()

    def _reset_sensor_segments(self) -> None:
        self._sensor_segments.first_points[:] = self._player_loc_np
        player_angle = self._player.angle
        self._end_biases_units = np.vstack((np.cos(self._angles + player_angle),
                                            np.sin(self._angles + player_angle))).T
        end_biases = self._end_biases_units * self._max_distance
        self._sensor_segments.second_points[:] = end_biases + self._player_loc_np

    def _init_segments(self) -> None:
        for i, wall in enumerate(self._walls):
            indices = np.arange(i * 4, i * 4 + 4)
            fp, sp = get_rect_segments_np(wall.rectangle)
            self._wall_segments.first_points[indices, :] = fp
            self._wall_segments.second_points[indices, :] = sp
        loop_array(self._wall_segments.first_points, len(self._walls) * 4)
        loop_array(self._wall_segments.second_points, len(self._walls) * 4)

        for i, chest in enumerate(self._gold_chests):
            indices = np.arange(i * 4, i * 4 + 4)
            fp, sp = get_rect_segments_np(chest.rectangle)
            self._chest_segments.first_points[indices, :] = fp
            self._chest_segments.second_points[indices, :] = sp
        loop_array(self._chest_segments.first_points, len(self._gold_chests) * 4)
        loop_array(self._chest_segments.second_points, len(self._gold_chests) * 4)

        indices = np.arange(4)
        fp, sp = get_rect_segments_np(self._portal.rectangle)
        self._portal_segments.first_points[indices, :] = fp
        self._portal_segments.second_points[indices, :] = sp
        loop_array(self._portal_segments.first_points, 4)
        loop_array(self._portal_segments.second_points, 4)

        self._reset_sensor_segments()
        self._points_np[:] = self._sensor_segments.second_points

    def _get_min_norms(self,
                       comparing_indices: np.ndarray,
                       other_segments: LineSegments,
                       n_segments: int,
                       active: Optional[np.ndarray] = None) -> np.ndarray:
        comparing_segments = LineSegments(
            self._sensor_segments.first_points[comparing_indices],
            self._sensor_segments.second_points[comparing_indices],
        )
        comparison = np_line_segments_intersect(comparing_segments, other_segments)
        intersects = comparison.intersect_indicators.reshape((self._n_sensors, n_segments))
        if active is not None:
            active_segments = np.tile(active, (4, 1)).T.flatten()
            intersects = np.logical_and(intersects, active_segments)
        points = comparison.points.reshape((self._n_sensors, n_segments, 2))
        norms = np.linalg.norm(points - self._player_loc_np, axis=2)
        norms[np.logical_not(intersects)] = np.inf
        min_norms = np.min(norms, axis=1)
        return min_norms

    def update(self) -> None:

        x, y = self._player.position
        self._player_loc_np[0] = x
        self._player_loc_np[1] = y
        self._reset_sensor_segments()
        self._current_objects[:] = SensedObject.NONE.value

        self._distances[:] = np.inf

        wall_comparing_indices = np.tile(np.arange(self._n_sensors), (4 * len(self._walls), 1)).T.flatten()
        wall_min_norms = self._get_min_norms(wall_comparing_indices, self._wall_segments, 4 * len(self._walls))
        no_wall = np.isinf(wall_min_norms)
        wall_found = np.logical_not(no_wall)
        self._distances[wall_found] = wall_min_norms[wall_found]
        self._current_objects[wall_found] = SensedObject.WALL.value

        chest_active = np.array([not chest.collected for chest in self._gold_chests], dtype=np.bool)
        chest_comparing_indices = np.tile(np.arange(self._n_sensors), (4 * len(self._gold_chests), 1)).T.flatten()
        chest_min_norms = self._get_min_norms(
            chest_comparing_indices, self._chest_segments, 4 * len(self._gold_chests), chest_active
        )
        no_chest = np.isinf(chest_min_norms)
        chest_found = np.logical_and(chest_min_norms < self._distances, np.logical_not(no_chest)),
        self._distances[chest_found] = chest_min_norms[chest_found]
        self._current_objects[chest_found] = SensedObject.GOLD.value

        portal_comparing_indices = np.tile(np.arange(self._n_sensors), (4, 1)).T.flatten()
        portal_min_norms = self._get_min_norms(portal_comparing_indices, self._portal_segments, 4)
        no_portal = np.isinf(portal_min_norms)
        portal_found = np.logical_and(portal_min_norms < self._distances, np.logical_not(no_portal))
        self._distances[portal_found] = portal_min_norms[portal_found]
        self._current_objects[portal_found] = SensedObject.PORTAL.value

        self._distances[np.isinf(self._distances)] = self._max_distance
        self._points_np = self._player_loc_np + self._end_biases_units * self._distances.reshape((self._n_sensors, 1))

    def reset(self) -> None:
        self._reset_sensor_segments()
        self._points_np[:] = self._sensor_segments.second_points
        self._current_objects[:] = SensedObject.NONE.value

    @property
    def points(self) -> np.ndarray:
        return self._points_np

    @property
    def distances(self):
        return self._distances

    @property
    def object_types(self) -> np.ndarray:
        return self._current_objects

    @property
    def n_sensors(self) -> int:
        return self._n_sensors

    @property
    def max_distance(self):
        return self._max_distance


class ProximitySensor:
    _player = ...  # type: Player
    _angle = ...  # type: float
    _max_distance = ...  # type: float
    _walls = ...  # type: List[Wall]
    _gold_chests = ...  # type: List[GoldChest]
    _portal = ...  # type: Portal

    _point = ...  # type: Vector2
    _current_obj = ...  # type: SensedObject
    _segment = ...  # type: LineSegment

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


class WallsCollisionChecker:
    _walls = ...  # type: List[RectangleWall]

    _wall_segments = ...  # type: LineSegments

    def __init__(self, walls: List[Wall]):
        self._walls = walls

        self._wall_segments = LineSegments(
            np.empty(shape=(len(self._walls) * 4, 2), dtype=np.float32),
            np.empty(shape=(len(self._walls) * 4, 2), dtype=np.float32),
        )

        for i, wall in enumerate(self._walls):
            indices = np.arange(i * 4, i * 4 + 4)
            fp, sp = get_rect_segments_np(wall.rectangle)
            self._wall_segments.first_points[indices, :] = fp
            self._wall_segments.second_points[indices, :] = sp

    def segment_collides_with_walls(self, line_segment: LineSegment) -> bool:
        line_segments = LineSegments(
            np.tile(np.array([line_segment.a.x, line_segment.a.y]), (len(self._walls) * 4, 1)),
            np.tile(np.array([line_segment.b.x, line_segment.b.y]), (len(self._walls) * 4, 1)),
        )
        return np.any(np_line_segments_intersect(line_segments, self._wall_segments).intersect_indicators)


class World:
    _width = ...  # type: float
    _height = ...  # type: float
    _player = ...  # type: Player
    _walls = ...  # type: List[Wall]
    _gold_chests = ...  # type: List[GoldChest]
    _heat_sources = ...  # type: List[HeatSource]
    _portal = ...  # type: Portal
    _validator = ...  # type: LocationValidator
    _game_over = ...  # type: bool
    _proximity_sensors_np = ...  # type: ProximitySensors
    _remember_position_interval_s = ...  # type: float
    _visit_reward_impact_decay_per_s = ...  # type: float
    _n_nearby_visit_points = ...  # type: int
    _visit_coef_ps = ...  # type: float

    _current_time_s = ...  # type: float
    _last_saved_position_time_s = ...  # type: float
    _player_visits = ...  # type: List[Tuple[float, Vector2]]

    _exploration_pressure = ...  # type: float
    _last_ep_update = ...  # type: float
    _update_ep_every_s = ...  # type: float

    _wall_collision_checker = ...  # type: WallsCollisionChecker

    def __init__(
            self,
            width: float,
            height: float,
            player: Player,
            walls: List[Wall],
            gold_chests: List[GoldChest],
            heat_sources: List[HeatSource],
            portal: Portal,
            proximity_sensors_np: ProximitySensors):
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

        self._proximity_sensors_np = proximity_sensors_np

        self._remember_position_interval_s = 0.35
        self._n_nearby_visit_points = 100
        # self._visit_reward_impact_decay_per_s = 0.987
        # self._visit_coef_ps = 3000.
        self._time_diff_coef = 0.1
        self._pos_diff_coef = 0.1

        self._current_time_s = 0.
        self._last_saved_position_time_s = 0.
        self._player_visits = []

        self._exploration_pressure = 0.
        self._last_ep_update = 0.
        self._update_ep_every_s = 0.35

        self._wall_collision_checker = WallsCollisionChecker(self._walls)

    def _update_player_angle(self, new_angle: float) -> None:
        if not self._game_over:
            self._player.update_angle(new_angle)

    def _move_player(self, elapsed_time_s: float, direction: PlayerMovementDirection) -> None:
        if self._game_over:
            return

        player_pos = self._player.position
        translation = self._player.get_translation(elapsed_time_s, direction)
        new_player_points = get_rectangle_points(
            make_rectangle(
                player_pos + translation,
                self._player.width,
                self._player.height
            )
        )
        old_player_points = get_rectangle_points(
            make_rectangle(
                player_pos,
                self._player.width,
                self._player.height
            )
        )
        for np, op in zip(new_player_points, old_player_points):
            if not self._validator.is_valid(Vector2(np.x, op.y)):
                translation.x = 0
            if not self._validator.is_valid(Vector2(op.x, np.y)):
                translation.y = 0
        self._player.apply_translation(translation)

    def _update_state(self, elapsed_time_s: float) -> None:
        if self._game_over:
            return

        portal_rect = self._portal.rectangle
        player_rect = self._player.get_rectangle()
        if rectangles_intersect(player_rect, portal_rect).intersects:
            self._game_over = True
            self._player.add_portal_reward()

        for gold_chest in self._gold_chests:
            if not gold_chest.collected and rectangles_intersect(gold_chest.rectangle, player_rect).intersects:
                self._player.add_gold(gold_chest.collect())

        player_pos = self._player.position
        heat = sum(hs.get_heat(player_pos) for hs in self._heat_sources)
        self._player.set_heat(heat)

        self._proximity_sensors_np.update()

        self._player.apply_passive_reward_penalty(elapsed_time_s)

        self._update_exploration_pressure(elapsed_time_s)

    def _update_player_positions(self) -> None:
        if self._current_time_s - self._last_saved_position_time_s < self._remember_position_interval_s:
            return
        self._last_saved_position_time_s = self._current_time_s
        self._player_visits.append((self._current_time_s, self._player.position))

    def _update_exploration_pressure(self, elapsed_time_s: float) -> None:
        if self._current_time_s - self._last_ep_update < self._update_ep_every_s:
            return

        self._last_ep_update = self._current_time_s

        current_position = self._player.position
        relevant_visits = list(
            sorted(
                (
                    (self._current_time_s - time, abs(current_position - visit_position))
                    for time, visit_position in self._player_visits
                    if (
                        current_position != visit_position
                        and not self._wall_collision_checker.segment_collides_with_walls(
                            LineSegment(current_position, visit_position)
                        )
                        # and self._current_time_s - time > 1.
                    )
                ),
                key=lambda t: t[1]
            )
        )[:self._n_nearby_visit_points]
        # print([time_diff for time_diff, _ in relevant_visits])
        # print([
        #     self._visit_reward_impact_decay_per_s ** time_diff
        #     for time_diff, pos_diff in relevant_visits
        # ])
        # reward_impact = sum(
        #     (self._visit_coef_ps * self._visit_reward_impact_decay_per_s ** time_diff / (
        #         pos_diff if pos_diff > 50 else 50
        #     ))
        #     for time_diff, pos_diff in relevant_visits
        # ) * elapsed_time_s / self._n_nearby_visit_points
        reward_impact = sum(
            time_diff * self._time_diff_coef + pos_diff * self._pos_diff_coef
            for time_diff, pos_diff in relevant_visits
        ) * elapsed_time_s / self._n_nearby_visit_points
        # print(reward_impact)
        self._exploration_pressure = reward_impact

    def update_world_and_player_and_get_reward(
            self,
            elapsed_time_s: float,
            new_angle: float,
            direction: PlayerMovementDirection
    ) -> float:
        self._update_player_angle(new_angle)
        self._move_player(elapsed_time_s, direction)
        self._update_state(elapsed_time_s)
        self._current_time_s += elapsed_time_s
        self._update_player_positions()
        self._player.add_custom_reward(self._exploration_pressure)

        reward = self._player.reward
        self._player.reset_reward_after_step()
        return reward

    def get_visible_visits(self) -> Set[int]:
        current_position = self._player.position
        return {
            i for i, (_, visit_pos) in enumerate(self._player_visits)
            if not self._wall_collision_checker.segment_collides_with_walls(LineSegment(current_position, visit_pos))
        }

    def reset(self) -> None:
        self._game_over = False
        self._player.reset()
        for chest in self._gold_chests:
            chest.reset()
        self._proximity_sensors_np.reset()
        self._player_visits = []
        self._current_time_s = 0.
        self._last_saved_position_time_s = 0.

    @property
    def player(self) -> Player:
        return self._player

    @property
    def width(self) -> float:
        return self._width

    @property
    def height(self) -> float:
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
    def proximity_sensors_np(self) -> ProximitySensors:
        return self._proximity_sensors_np

    @property
    def player_visits(self):
        return self._player_visits

    @property
    def exploration_pressure(self):
        return self._exploration_pressure

    @property
    def current_time_s(self):
        return self._current_time_s
