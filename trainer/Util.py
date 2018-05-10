from collections import namedtuple
from math import sin, cos, sqrt
from typing import NamedTuple, Optional, Union

import numpy as np


class Vector2:
    x = ...  # type: float
    y = ...  # type: float

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def clone(self):
        return Vector2(self.x, self.y)

    def __abs__(self):
        return sqrt(self.x ** 2 + self.y ** 2)

    def __mul__(self, scalar: float):
        return Vector2(self.x * scalar, self.y * scalar)

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def __isub__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def __iter__(self):
        return iter((self.x, self.y))

    def __str__(self):
        return "({x}, {y})".format(x=self.x, y=self.y)

    def __repr__(self):
        return "({x}, {y})".format(x=self.x, y=self.y)


def dot(v1: Vector2, v2: Vector2) -> float:
    return v1.x * v2.x + v1.y * v2.y


def dots(v1s: np.ndarray, v2s: np.ndarray) -> np.ndarray:
    return v1s[:, 0] * v2s[:, 0] + v1s[:, 1] + v2s[:, 1]


def cross(v1: Vector2, v2: Vector2) -> float:
    return v1.x * v2.y - v1.y * v2.x


def crosses(v1s: np.ndarray, v2s: np.ndarray) -> np.ndarray:
    return v1s[:, 0] * v2s[:, 1] - v1s[:, 1] * v2s[:, 0]


Ray = namedtuple("Ray", [
    "origin",
    "angle"
])
# class Ray(NamedTuple):
#     origin: Vector2
#     angle: float


LineSegment = namedtuple("LineSegment", [
    "a",
    "b"
])
# class LineSegment(NamedTuple):
#     a: Vector2
#     b: Vector2


RectangleAABB = namedtuple("RectangleAABB", [
    "lower_left",
    "upper_right"
])
# class RectangleAABB(NamedTuple):
#     lower_left: Vector2
#     upper_right: Vector2


RectanglePoints = namedtuple("RectanglePoints", [
    "lower_left",
    "lower_right",
    "upper_right",
    "upper_left"
])
# class RectanglePoints(NamedTuple):
#     lower_left: Vector2
#     lower_right: Vector2
#     upper_right: Vector2
#     upper_left: Vector2


def make_rectangle(center: Vector2, width: float, height: float) -> RectangleAABB:
    return RectangleAABB(
        Vector2(center.x - width, center.y - height),
        Vector2(center.x + width, center.y + height)
    )


def to_the_left(point: Vector2, p1: Vector2, p2: Vector2) -> bool:
    a = -(p2.y - p1.y)
    b = p2.x - p1.x
    c = -(a * p1.x + b * p1.y)
    return a * point.x + b * point.y + c > 0


def get_rectangle_points(rectangle: RectangleAABB) -> RectanglePoints:
    ll, ur = rectangle
    height = ur.y - ll.y
    width = ur.x - ll.x
    ul = Vector2(ll.x, ll.y + height)
    lr = Vector2(ll.x + width, ll.y)
    return RectanglePoints(ll, lr, ur, ul)


def point_in_rectangle_points(point: Vector2, points: RectanglePoints) -> bool:
    return (
            to_the_left(point, points.lower_left, points.lower_right) and
            to_the_left(point, points.lower_right, points.upper_right) and
            to_the_left(point, points.upper_right, points.upper_left) and
            to_the_left(point, points.upper_left, points.lower_left)
    )


def point_in_rectangle_aabb(point: Vector2, rectangle: RectangleAABB) -> bool:
    return (
            rectangle.lower_left.x < point.x < rectangle.upper_right.x and
            rectangle.lower_left.y < point.y < rectangle.upper_right.y
    )


RectangleCollision = namedtuple("RectangleCollision", [
    "intersects",
    "point"
])
# class RectangleCollision(NamedTuple):
#     intersects: bool
#     point: Optional[Vector2]


def rectangles_intersect(r1: RectangleAABB, r2: RectangleAABB) -> RectangleCollision:
    for point in get_rectangle_points(r1):
        if point_in_rectangle_aabb(point, r2):
            return RectangleCollision(True, point)
    return RectangleCollision(False, None)


Collision = namedtuple("Collision", [
    "intersects",
    "point"
])
# class Collision(NamedTuple):
#     intersects: bool
#     point: Optional[Vector2]


def line_segments_intersect(s1: LineSegment, s2: LineSegment) -> Collision:
    p = s1.a
    r = s1.b - s1.a
    q = s2.a
    s = s2.b - s2.a

    cross_r_s = cross(r, s)
    cross_q_p_r = cross(q - p, r)

    if cross_r_s == 0:
        if cross_q_p_r == 0:
            t0 = dot(q - p, r) / dot(r, r)
            t1 = dot(q + s - p, r) / dot(r, r)
            if t1 <= 0 <= t0 or t1 <= 1 <= t0 or 0 <= t1 <= t0 <= 1:
                raise ArithmeticError("oh no")
            else:
                return Collision(False, None)
        return Collision(False, None)

    t = cross(q - p, s) / cross_r_s
    u = cross_q_p_r / cross_r_s

    intersects = 0 <= t <= 1 and 0 <= u <= 1
    point = p + r * t if intersects else None

    return Collision(intersects, point)


LineSegments = namedtuple("LineSegments", [
    "first_points",
    "second_points"
])
# class LineSegments(NamedTuple):
#     first_points: np.ndarray
#     second_points: np.ndarray


Collisions = namedtuple("Collisions", [
    "intersect_indicators",
    "points"
])
# class Collisions(NamedTuple):
#     intersect_indicators: np.ndarray
#     points: np.ndarray


def np_line_segments_intersect(s1: LineSegments, s2: LineSegments) -> Collisions:
    assert s1.first_points.shape == s1.second_points.shape == s2.first_points.shape == s2.second_points.shape, (s1.first_points.shape, s1.second_points.shape, s2.first_points.shape, s2.second_points.shape)

    p = s1.first_points
    r = s1.second_points - s1.first_points
    q = s2.first_points
    s = s2.second_points - s2.first_points

    output_shape = (s1.first_points.shape[0], 2)
    cross_r_s = np.reshape(crosses(r, s), (output_shape[0],))
    cross_q_p_r = np.reshape(crosses(q - p, r), (output_shape[0],))
    cross_q_p_s = np.reshape(crosses(q - p, s), (output_shape[0],))

    intersect_indicators = np.zeros(shape=(output_shape[0],), dtype=np.bool)
    points = np.zeros(shape=output_shape, dtype=np.float32)

    normal_indices = cross_r_s != 0

    normal_r_s = cross_r_s[normal_indices]
    normal_q_p_r = cross_q_p_r[normal_indices]
    normal_q_p_s = cross_q_p_s[normal_indices]
    t = normal_q_p_s / normal_r_s
    u = normal_q_p_r / normal_r_s

    sub_intersecting = np.logical_and(
        np.logical_and(0 <= t, t <= 1),
        np.logical_and(0 <= u, u <= 1)
    )
    intersect_indicators[normal_indices] = sub_intersecting
    if np.any(intersect_indicators):
        t_s = t[sub_intersecting]
        points[intersect_indicators] = p[intersect_indicators] + r[intersect_indicators] * t_s.reshape((t_s.size, 1))

    # cross_r_s_zero = np.logical_not(normal_indices)
    # still_normal = np.logical_and(cross_r_s_zero, cross_q_p_r == 0)
    # q_sn = q[still_normal]
    # p_sn = p[still_normal]
    # s_sn = s[still_normal]
    # r_sn = r[still_normal]
    # t0 = (dots(q_sn - p_sn, r_sn) / dots(r_sn, r_sn)).flatten()
    # t1 = (dots(q_sn + s_sn - p_sn, r_sn) / dots(r_sn, r_sn)).flatten()
    # absolutely_wrong = np.logical_or(
    #     np.logical_or(
    #         np.logical_and(t1 <= 0, 0 <= t0),
    #         np.logical_and(t1 <= 1, 1 <= t0)
    #     ),
    #     np.logical_and(
    #         np.logical_and(0 <= t1, t1 <= t0),
    #         t0 <= 1
    #     )
    # )
    # if np.any(absolutely_wrong):
    #     raise ArithmeticError("oh no")

    return Collisions(intersect_indicators, points)


def ray_segment_intersect(ray: Ray, a: Vector2, b: Vector2) -> Collision:
    o = ray.origin
    d = Vector2(cos(ray.angle), sin(ray.angle))
    c = b - a

    t2 = 1 / (d.y * c.x / d.x - c.y) * (a.y + d.y * o.x / d.x - d.y * a.x / d.x - o.y)
    t1 = (a.x + c.x * t2 - o.x) / d.x

    intersects = t1 >= 0 and 0 <= t2 <= 1
    point = ray.origin + d * t1 if intersects else None

    return Collision(intersects, point)


def ray_rectangle_aabb_intersect(ray: Ray, rectangle: RectangleAABB) -> Collision:
    ll, lr, ur, ul = get_rectangle_points(rectangle)

    min_collision = Collision(False, None)
    min_distance = float("inf")
    for a, b in [(ll, lr), (lr, ur), (ur, ul), (ul, ll)]:
        current_collision = ray_segment_intersect(ray, a, b)
        if current_collision.intersects:
            distance = abs(current_collision.point - ray.origin)
            if distance < min_distance:
                min_collision = current_collision
                min_distance = distance

    return min_collision


def segment_aabb_intersect(segment: LineSegment, aabb: RectangleAABB) -> Collision:
    ll, lr, ur, ul = get_rectangle_points(aabb)

    min_collision = Collision(False, None)
    min_distance = float("inf")
    for a, b in [(ll, lr), (lr, ur), (ur, ul), (ul, ll)]:
        current_collision = line_segments_intersect(segment, LineSegment(a, b))
        if current_collision.intersects:
            distance = abs(current_collision.point - segment.a)
            if distance < min_distance:
                min_collision = current_collision
                min_distance = distance

    return min_collision
