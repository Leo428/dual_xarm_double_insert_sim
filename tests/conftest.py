"""Shared test setup.

The sim env always constructs a MuJoCo ``Renderer``, which needs an OpenGL backend.
Default to EGL (headless GPU) unless the caller already chose one, so the tests run on
servers without an X display.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
