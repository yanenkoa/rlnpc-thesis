from math import sin, cos, sqrt
from typing import NamedTuple, Optional


class Vector2:
    x: float
    y: float

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

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
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"({self.x}, {self.y})"


def dot(v1: Vector2, v2: Vector2) -> float:
    return v1.x * v2.x + v1.y * v2.y


def cross(v1: Vector2, v2: Vector2) -> float:
    return v1.x * v2.y - v1.y * v2.x


class Ray(NamedTuple):
    origin: Vector2
    angle: float


class LineSegment(NamedTuple):
    a: Vector2
    b: Vector2


class RectangleAABB(NamedTuple):
    lower_left: Vector2
    upper_right: Vector2

    def __str__(self):
        return f"lower_left={self.lower_left}, upper_right={self.upper_right}"


class RectanglePoints(NamedTuple):
    lower_left: Vector2
    lower_right: Vector2
    upper_right: Vector2
    upper_left: Vector2


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


class RectangleCollision(NamedTuple):
    intersects: bool
    point: Optional[Vector2]


def rectangles_intersect(r1: RectangleAABB, r2: RectangleAABB) -> RectangleCollision:
    for point in get_rectangle_points(r1):
        if point_in_rectangle_aabb(point, r2):
            return RectangleCollision(True, point)
    return RectangleCollision(False, None)


class Collision(NamedTuple):
    intersects: bool
    point: Optional[Vector2]


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
