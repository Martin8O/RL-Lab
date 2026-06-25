"""The shared off-policy resume guard (SAC / TD3 / DQN).

On resume the replay buffer is empty (SB3 excludes it from ``model.zip``) yet the restored
``num_timesteps`` already exceeds ``learning_starts``, so SB3 would run gradient updates immediately on
the near-empty buffer and degrade the restored policy (the "load a 56 % save, watch it slide to 31 %"
bug). :class:`ResumeBufferGate` holds updates until the buffer refills. These tests drive the guard with
a fake algorithm so they stay fast + deterministic (the real SB3 resume path is already covered by each
algo's save→resume test).

The guard is a **mixin method**, not a monkeypatched ``model.train`` attribute, precisely so SB3's
``save`` (which serializes ``__dict__``) never pickles a closure that drags the whole model + replay
buffer into the blob (a real run OOMs). ``test_gate_is_not_in_instance_dict`` locks that in.
"""

from app.services.offpolicy import ResumeBufferGate


class _FakeBuffer:
    def __init__(self) -> None:
        self._size = 0

    def size(self) -> int:
        return self._size


class _FakeAlgo:
    """A stand-in for an SB3 off-policy algorithm: a growable buffer + a counting ``train``."""

    def __init__(self) -> None:
        self.replay_buffer = _FakeBuffer()
        self.train_calls = 0
        self.last_args: tuple = ()
        self.last_kwargs: dict = {}

    def train(self, *args: object, **kwargs: object) -> str:
        self.train_calls += 1
        self.last_args, self.last_kwargs = args, kwargs
        return "trained"


class _GatedAlgo(ResumeBufferGate, _FakeAlgo):
    """The mixin in front of the fake algorithm — exactly the shape `_ResumeSAC(ResumeBufferGate, SAC)`."""


def test_gate_skips_updates_until_buffer_reaches_threshold() -> None:
    m = _GatedAlgo()
    m.grad_start_size = 100

    # Below the threshold: train() is a no-op — the underlying update never runs (SB3 train returns None
    # either way, so the observable signal is whether the wrapped train was reached).
    m.replay_buffer._size = 0
    m.train()
    m.replay_buffer._size = 99
    m.train()
    assert m.train_calls == 0

    # At/above the threshold: train() passes through to the real implementation.
    m.replay_buffer._size = 100
    m.train()
    m.replay_buffer._size = 500
    m.train(gradient_steps=1, batch_size=64)
    assert m.train_calls == 2


def test_gate_forwards_args_to_the_wrapped_train() -> None:
    m = _GatedAlgo()
    m.grad_start_size = 10
    m.replay_buffer._size = 50
    m.train(gradient_steps=3, batch_size=128)
    assert m.last_kwargs == {"gradient_steps": 3, "batch_size": 128}


def test_gate_is_a_noop_when_threshold_zero() -> None:
    # grad_start_size defaults to 0 → never gates (a fresh run never sets it).
    m = _GatedAlgo()
    assert m.grad_start_size == 0
    m.replay_buffer._size = 0
    m.train()
    assert m.train_calls == 1


def test_gate_is_not_in_instance_dict() -> None:
    # The guard must be a class method, NOT an instance attribute: SB3's save() serializes __dict__, so a
    # closure there would drag the whole model (replay buffer included) into the blob → MemoryError.
    m = _GatedAlgo()
    m.grad_start_size = 100
    assert "train" not in m.__dict__  # train lives on the class via the mixin, never on the instance
