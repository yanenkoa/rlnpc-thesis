from typing import NamedTuple


class Vector2:

    x: float
    y: float

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

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


class Rectangle(NamedTuple):
    lower_left: Vector2
    upper_right: Vector2


class RectanglePoints(NamedTuple):
    lower_left: Vector2
    lower_right: Vector2
    upper_right: Vector2
    upper_left: Vector2


def make_rectangle(center: Vector2, width: float, height: float) -> Rectangle:
    return Rectangle(
        Vector2(center.x - width, center.y - height),
        Vector2(center.x + width, center.y + height)
    )


def to_the_left(point: Vector2, p1: Vector2, p2: Vector2) -> bool:
    a = -(p2.y - p1.y)
    b = p2.x - p1.x
    c = -(a * p1.x + b * p1.y)
    return a * point.x + b * point.y + c > 0


def get_rectangle_points(rectangle: Rectangle) -> RectanglePoints:
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


def rectangles_intersect(r1: Rectangle, r2: Rectangle) -> bool:
    points1 = get_rectangle_points(r1)
    points2 = get_rectangle_points(r2)
    return (
            point_in_rectangle_points(points1.lower_left,  points2) or
            point_in_rectangle_points(points1.lower_right, points2) or
            point_in_rectangle_points(points1.upper_right, points2) or
            point_in_rectangle_points(points1.upper_left,  points2)
    )
