import tkinter as tk
from math import sin, cos
from typing import Any, Tuple, Dict

from trainer.GameObjects import World, RectangleWall, GoldChest, Wall, HeatSource, SensedObject
from trainer.Util import Vector2, RectangleAABB, make_rectangle

UnpackedRectangle = Tuple[float, float, float, float]


def invert_y(v: Vector2, y_cap: float) -> Vector2:
    return Vector2(v.x, y_cap - v.y)


def invert_y_rectangle(r: RectangleAABB, y_cap: float) -> RectangleAABB:
    return RectangleAABB(invert_y(r.lower_left, y_cap), invert_y(r.upper_right, y_cap))


def unpack_rectangle(r: RectangleAABB) -> UnpackedRectangle:
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


def get_player_text(world: World):
    player = world.player
    return (
        (
            'Gold: {gold}\n'
            'Heat: {heat}\n'
            'Reward: {reward_sum}\n'
            '{game_over}'
        ).format(
            gold=player.gold,
            heat=player.heat,
            reward_sum=player.reward_sum,
            game_over=("Game over!" if world.game_over else "")
        )
    )


class RenderWorld:
    _player_color = "green"  # type: str
    _angle_marker_bias = 30.  # type: float
    _angle_marker_width = 5.  # type: float
    _angle_marker_height = 5.  # type: float
    _angle_marker_color = "blue"  # type: str
    _text_x = 100.  # type: float
    _text_y = 100.  # type: float

    _world = ...  # type: World
    _started = ...  # type: bool
    _master = ...  # type: tk.Tk
    _canvas = ...  # type: tk.Canvas
    _player_fig = ...  # type: Any
    _angle_marker_fig = ...  # type: Any
    _walls_figs = ...  # type: Dict[Wall, Any]
    _gold_chest_figs = ...  # type: Dict[GoldChest, Any]
    _heat_source_figs = ...  # type: Dict[HeatSource, Any]
    _prox_sens_figs = ...  # type: Dict[int, Any]

    def __init__(self, world: World):

        self._started = False
        self._world = world

        self._master = tk.Tk()

        self._canvas = tk.Canvas(self._master, width=world.width, height=world.height)
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
            tag="portal_fig"
        )

        self._walls_figs = {}
        for i, wall in enumerate(self._world.walls):
            if isinstance(wall, RectangleWall):
                wall_fig = self._canvas.create_rectangle(
                    *unpack_rectangle(invert_y_rectangle(wall.rectangle, self._world.height)),
                    fill="black",
                    width=0,
                    tag="wall_fig_{i}".format(i=i)
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
                tag="gold_chest_fig_{i}".format(i=i)
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
                tag="heat_source_fig{i}".format(i=i)
            )
            self._heat_source_figs[heat_source] = heat_source_fig

        self._text = self._canvas.create_text(self._text_x, self._text_y, text=str(get_player_text(self._world)))

        self._prox_sens_figs = {}
        points = self._world.proximity_sensors_np.points
        for i in range(points.shape[0]):
            prox_sens_fig = self._canvas.create_oval(
                *make_unpacked_inverted_rectangle(
                    Vector2(points[i, 0], points[i, 1]),
                    5,
                    5,
                    self._world.height
                ),
                fill="grey",
                width=1,
                tag="prox_sens_fig{i}".format(i=i)
            )
            self._prox_sens_figs[i] = prox_sens_fig

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

    def start_drawing(self) -> None:
        if not self._started:
            self._started = True
            self._canvas.pack()

    def _redraw(self) -> None:
        self._canvas.coords(self._player_fig, *self._get_player_fig_coords())
        self._canvas.coords(self._angle_marker_fig, *self._get_angle_marker_coords())
        text = str(get_player_text(self._world))
        self._canvas.itemconfig(self._text, text=text)
        for i, gold_chest in enumerate(self._world.gold_chests):
            if gold_chest.collected and gold_chest in self._gold_chest_figs:
                self._canvas.delete(self._gold_chest_figs[gold_chest])
                self._gold_chest_figs.pop(gold_chest)
            if not gold_chest.collected and gold_chest not in self._gold_chest_figs:
                gold_chest_fig = self._canvas.create_rectangle(
                    *make_unpacked_inverted_rectangle(
                        gold_chest.location, gold_chest.width, gold_chest.height, self._world.height
                    ),
                    fill="yellow",
                    width=1,
                    tag="gold_chest_fig_{i}".format(i=i)
                )
                self._gold_chest_figs[gold_chest] = gold_chest_fig
        points = self._world.proximity_sensors_np.points
        senses = self._world.proximity_sensors_np.object_types
        for i in range(points.shape[0]):
            self._canvas.coords(
                self._prox_sens_figs[i],
                make_unpacked_inverted_rectangle(
                    Vector2(points[i, 0], points[i, 1]),
                    5,
                    5,
                    self._world.height
                )
            )
            if senses[i] == SensedObject.NONE.value:
                color = "grey"
            elif senses[i] == SensedObject.WALL.value:
                color = "black"
            elif senses[i] == SensedObject.GOLD.value:
                color = "yellow"
            elif senses[i] == SensedObject.PORTAL.value:
                color = "blue"
            else:
                raise ValueError("sensed_obj is fucked up")
            self._canvas.itemconfig(self._prox_sens_figs[i], fill=color)

    def update(self):
        if not self._started:
            return
        self._redraw()
        self._master.update_idletasks()
        self._master.update()
