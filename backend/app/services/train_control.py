"""Thread-control primitive shared by the training manager and the SB3 callback.

Kept free of any ML imports so the manager can create/inspect it without pulling in
torch/SB3. The SB3 callback parks on this between env steps to implement pause/stop.
"""

import threading


class TrainControl:
    """Cooperative pause/stop signalling for a background training run.

    The trainer's callback calls :meth:`wait_if_paused` and checks :attr:`stop_requested`
    on every step. ``_resume`` is *set* while running and *cleared* while paused, so a
    paused worker thread blocks inside ``wait()`` until resumed or stopped.
    """

    def __init__(self) -> None:
        self.stop_requested = False
        self._resume = threading.Event()
        self._resume.set()  # start in the running (non-paused) state

    @property
    def paused(self) -> bool:
        return not self._resume.is_set()

    def pause(self) -> None:
        self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    def request_stop(self) -> None:
        self.stop_requested = True
        self._resume.set()  # unblock a paused worker so it can observe the stop

    def wait_if_paused(self) -> None:
        self._resume.wait()
