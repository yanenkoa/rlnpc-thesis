import tkinter as tk
from math import pi, sin, cos
from typing import Any, Tuple, Dict

from pynput.keyboard import Key

from GameObjects import World, Player, RectangleConstraints, RectangleWall, GoldChest, Wall, HeatSource, Portal
from InputDevice import InputDevice
from Util import Vector2, Rectangle, make_rectangle

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


def get_player_text(player: Player):
    return f"Gold: {player.gold}\nHeat: {player.heat}"


class RenderWorld:

    _player_color: str = "green"
    _angle_marker_bias: float = 30
    _angle_marker_width: float = 5
    _angle_marker_height: float = 5
    _angle_marker_color: str = "blue"
    _text_x: float = 100
    _text_y: float = 100

    _world: World
    _master: tk.Tk
    _input_device: InputDevice

    _canvas: tk.Canvas
    _player_fig: Any
    _angle_marker_fig: Any
    _walls_figs: Dict[Wall, Any]
    _gold_chest_figs: Dict[GoldChest, Any]
    _heat_source_figs: Dict[HeatSource, Any]

    def __init__(self, world: World, master: tk.Tk, input_device: InputDevice):

        self._world = world
        self._master = master
        self._input_device = input_device

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
        self._portal_fig = self._canvas.create_rectangle(
            *make_unpacked_inverted_rectangle(
                self._world.portal.location,
                self._world.portal.width,
                self._world.portal.height,
                self._world.height
            ),
            fill="blue",
            width=1,
            tag=f"portal_fig"
        )

        self._walls_figs = {}
        for i, wall in enumerate(self._world.walls):
            if isinstance(wall, RectangleWall):
                wall_fig = self._canvas.create_rectangle(
                    *unpack_rectangle(invert_y_rectangle(wall.rectangle, self._world.height)),
                    fill="black",
                    width=0,
                    tag=f"wall_fig_{i}"
                )
                self._walls_figs[wall] = wall_fig

        self._gold_chest_figs = {}
        for i, gold_chest in enumerate(self._world.gold_chests):
            gold_chest_fig = self._canvas.create_rectangle(
                *make_unpacked_inverted_rectangle(
                    gold_chest.location, gold_chest.width, gold_chest.height, self._world.height
                ),
                fill="yellow",
                width=1,
                tag=f"gold_chest_fig_{i}"
            )
            self._gold_chest_figs[gold_chest] = gold_chest_fig

        self._heat_source_figs = {}
        for i, heat_source in enumerate(self._world.heat_sources):
            heat_source_fig = self._canvas.create_oval(
                *make_unpacked_inverted_rectangle(
                    heat_source.location, heat_source.radius, heat_source.radius, self._world.height
                ),
                fill="orange",
                width=1,
                tag=f"heat_source_fig{i}"
            )
            self._heat_source_figs[heat_source] = heat_source_fig

        self._text = self._canvas.create_text(self._text_x, self._text_y, text=str(get_player_text(self._world.player)))
        self._canvas.pack()

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
        text = str(get_player_text(self._world.player))
        if self._world.game_over:
            text += "\nGame over!"
        self._canvas.itemconfig(self._text, text=text)
        for gold_chest in self._world.gold_chests:
            if gold_chest.collected:
                self._canvas.delete(self._gold_chest_figs[gold_chest])

    def loop(self):
        while True:
            self._world.update()
            self._update()
            self._master.update_idletasks()
            self._master.update()
            if self._input_device.is_key_down(Key.esc):
                break


def main():
    width = 1000
    height = 1000
    input_device = InputDevice()
    strategy = RectangleConstraints(width, height)
    p = Player(Vector2(width - 50, 50), pi / 2, 300, pi / 0.8, input_device)
    walls = [
        RectangleWall(Rectangle(Vector2(800,   0), Vector2(900, 400))),
        RectangleWall(Rectangle(Vector2(500, 400), Vector2(900, 500))),
        RectangleWall(Rectangle(Vector2(500,   0), Vector2(600, 300))),
        RectangleWall(Rectangle(Vector2(100,   0), Vector2(200, 300))),
        RectangleWall(Rectangle(Vector2(200, 100), Vector2(400, 200))),
        RectangleWall(Rectangle(Vector2(100, 300), Vector2(300, 400))),
        RectangleWall(Rectangle(Vector2(100, 300), Vector2(300, 400))),
        RectangleWall(Rectangle(Vector2(600, 600), Vector2(950, 700))),
        RectangleWall(Rectangle(Vector2(850, 700), Vector2(950, 950))),
        RectangleWall(Rectangle(Vector2(600, 700), Vector2(700, 850))),
        RectangleWall(Rectangle(Vector2(0,   700), Vector2(450, 800))),
    ]
    gold_chests = [
        GoldChest(500, Vector2(50,  50)),
        GoldChest(300, Vector2(250, 50)),
        GoldChest(100, Vector2(750, 50)),
        GoldChest(50,  Vector2(350, 450)),
        GoldChest(200, Vector2(50,  850)),
    ]
    heat_sources = [
        HeatSource(1000, Vector2(450, 450)),
    ]
    w = World(width, height, p, walls, gold_chests, heat_sources, Portal(Vector2(800, 750)))
    m = tk.Tk()
    rw = RenderWorld(w, m, input_device)
    rw.loop()


if __name__ == '__main__':
    main()
