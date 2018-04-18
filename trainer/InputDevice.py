from typing import Set, Union

from pynput.keyboard import Listener, Key, KeyCode

KeyType = Union[Key, KeyCode, str]


class InputDevice:
    _pressed_keys = ...  # type: Set[KeyType]
    _listener = ...  # type: Listener

    def __init__(self):
        self._pressed_keys = set()
        self._listener = Listener(
            on_press=self._on_press,
            on_release=self._on_release)
        self._listener.start()

    def _on_press(self, key: KeyType) -> None:
        self._pressed_keys.add(key)

    def _on_release(self, key: KeyType) -> None:
        self._pressed_keys.remove(key)

    def is_key_down(self, key: KeyType) -> bool:
        key = key if isinstance(key, KeyCode) or isinstance(key, Key) else KeyCode(char=key)
        return key in self._pressed_keys

    def stop(self) -> None:
        self._listener.stop()
