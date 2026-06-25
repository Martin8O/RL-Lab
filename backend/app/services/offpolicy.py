"""Shared helpers for the off-policy trainers (SAC / TD3 / DQN).

These three algorithms learn from a **replay buffer** and share the same resume hazard, so the guard
lives here once rather than being copy-pasted into each near-identical trainer.

The empty-buffer-on-resume bug
------------------------------
SB3 excludes the replay buffer from ``model.zip`` (it can be gigabytes), so a *resumed* off-policy run
starts with an **empty** buffer. SB3 ties both random warmup *and* the gradient-update gate to a single
``learning_starts`` threshold: while ``num_timesteps < learning_starts`` it plays random actions and
does no updates; above it, it plays the policy and updates every step. On resume the restored
``num_timesteps`` (say 330k) already far exceeds the saved ``learning_starts`` (say 10k), so SB3 begins
**gradient updates immediately on the near-empty buffer** — sampling a 256-row minibatch from a buffer
holding a handful of transitions means training on the same few rows over and over, which overfits the
critic/Q-net and *degrades the restored policy*. Symptom (reported on a SAC Humanoid resume): load a
56 %-skill save, and within ~1.5 min the score slides to ~31 % and stays there.

The fix (:class:`ResumeBufferGate`)
-----------------------------------
Mix this into the SB3 algorithm class on the **resume** path and set ``grad_start_size``: ``train`` then
no-ops until the buffer has refilled to that size (the same warmup a fresh run uses). Crucially,
**collection is not gated** — because ``num_timesteps`` already exceeds ``learning_starts``, SB3 collects
with the *restored policy* (not random), so the buffer refills with good on-policy data and the agent
keeps playing at its saved level during the refill. Once the buffer is full enough, updates resume on a
diverse batch and the policy continues from where it left off instead of collapsing.

Why a mixin (a real method) and **not** a monkeypatched ``model.train``: SB3's ``save`` serializes the
instance ``__dict__``. A closure assigned to ``model.train`` lands in ``__dict__`` and captures the whole
model — replay buffer included — so the first snapshot ``model.save()`` tries to pickle gigabytes and
dies with ``MemoryError``. A method defined on a class is never in ``__dict__``, so ``save`` ignores it;
the only instance attribute added is ``grad_start_size`` (a plain int, harmless in the blob).

This only touches the resume path; a fresh run uses the bare SB3 class and is byte-for-byte unchanged.
"""

from typing import Any


class ResumeBufferGate:
    """Mixin: hold gradient updates until the replay buffer holds ``grad_start_size`` transitions.

    Mix in **before** the SB3 algorithm class (``class _ResumeSAC(ResumeBufferGate, SAC)``) so this
    ``train`` overrides the algorithm's via the MRO and ``super().train`` reaches the real one. Default
    ``grad_start_size`` is 0 (never gates); the resume path sets it after ``load``.
    """

    grad_start_size: int = 0

    def train(self, *args: Any, **kwargs: Any) -> None:
        rb = getattr(self, "replay_buffer", None)
        if self.grad_start_size > 0 and rb is not None and rb.size() < self.grad_start_size:
            return  # buffer still refilling — skip the gradient update this step
        super().train(*args, **kwargs)  # type: ignore[misc]
