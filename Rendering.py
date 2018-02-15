from math import pi, sin, cos
import tkinter as tk
from typing import Any, Tuple

from GameObjects import World, Player, RectangleConstraints, WallType, RectangleWall
from InputDevice import InputDevice
from Util import Vector2, Rectangle, make_rectangle

import time

UnpackedRectangle = Tuple[float, float, float, float]


def invert_y(v: Vector2, y_cap: float) -> Vector2:
    return Vector2(v.x, y_cap - v.y)


def invert_y_rectangle(r: Rectangle, y_cap: float) -> Rectangle:
    return Rectangle(invert_y(r.lower_left, y_cap), invert_y(r.upper_right, y_cap))


def unpack_rectangle(r: Rectangle) -> UnpackedRectangle:
    ll, ur = r
    xll, yll = ll
    xur, yur = ur
    return xll, yll, xur, yur


def make_unpacked_inverted_rectangle(center: Vector2, width, height: float, y_cap: float) -> UnpackedRectangle:
    return unpack_rectangle(
        invert_y_rectangle(
            make_rectangle(center, width, height),
            y_cap
        )
    )


class RenderWorld:
    _player_color: str = "green"
    _angle_marker_bias: float = 30
    _angle_marker_width: float = 5
    _angle_marker_height: float = 5
    _angle_marker_color: str = "blue"

    _master: tk.Tk
    _world: World
    _canvas: tk.Canvas
    _player_fig: Any
    _angle_marker_fig: Any
    _walls_figs: Any

    def __init__(self, world: World, master: tk.Tk):
        self._master = master
        self._world = world
        self._canvas = tk.Canvas(master, width=world.width, height=world.height)
        self._canvas.create_rectangle(
            0, 0, self._world.width, self._world.height,
            fill="white",
            width=0
        )
        self._player_fig = self._canvas.create_oval(
            *self._get_player_fig_coords(),
            fill=self._player_color,
            width=1,
            tag="player"
        )
        self._angle_marker_fig = self._canvas.create_oval(
            *self._get_angle_marker_coords(),
            fill=self._angle_marker_color,
            width=1,
            tag="angle_marker"
        )
        self._walls_figs = []
        for i, wall in enumerate(self._world.walls):
            if isinstance(wall, RectangleWall):
                wall_fig = self._canvas.create_rectangle(
                    *unpack_rectangle(invert_y_rectangle(wall.rectangle, self._world.height)),
                    fill="black",
                    width=0,
                    tag=f"wall_fig_{i}"
                )
                self._walls_figs.append(wall_fig)
        self._canvas.pack()
        self._text = self._canvas.create_text(100, 100, text="0")

    def _get_player_fig_coords(self) -> UnpackedRectangle:
        return make_unpacked_inverted_rectangle(
            self._world.player.position,
            self._world.player.width,
            self._world.player.height,
            self._world.height
        )

    def _get_angle_marker_coords(self) -> UnpackedRectangle:
        x_player, y_player = self._world.player.position
        angle_player = self._world.player.angle
        x_bias = self._angle_marker_bias * cos(angle_player)
        y_bias = self._angle_marker_bias * sin(angle_player)
        x_center = x_player + x_bias
        y_center = y_player + y_bias
        return make_unpacked_inverted_rectangle(
            Vector2(x_center, y_center),
            self._angle_marker_width,
            self._angle_marker_height,
            self._world.height
        )

    def _update(self) -> None:
        self._canvas.coords(self._player_fig, *self._get_player_fig_coords())
        self._canvas.coords(self._angle_marker_fig, *self._get_angle_marker_coords())
        self._canvas.itemconfigure(self._text, text=str(time.time()))

    def loop(self):
        while True:
            self._world.update()
            self._update()
            self._master.update_idletasks()
            self._master.update()


def main():
    width = 1000
    height = 1000
    input_device = InputDevice()
    strategy = RectangleConstraints(width, height)
    p = Player(Vector2(width - 50, 50), 300, pi / 0.8, input_device)
    walls = [
        RectangleWall(Rectangle(Vector2(800, 0), Vector2(900, 400))),
        RectangleWall(Rectangle(Vector2(500, 400), Vector2(900, 500)))
    ]
    w = World(width, height, p, walls)
    m = tk.Tk()
    rw = RenderWorld(w, m)
    rw.loop()


if __name__ == '__main__':
    main()
