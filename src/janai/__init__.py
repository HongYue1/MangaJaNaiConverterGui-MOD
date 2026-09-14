"""JaNai Upscaler.

Three packages, deliberately separated:

``janai.app``
    The Qt (PySide6) interface. It never imports torch, numpy or pyvips, which
    is what keeps the window responsive and the startup instant.

``janai.core``
    Everything both processes need to agree on: where files live, the output
    format catalogue, the display presets, the upscaling rules and presets.

``janai.worker``
    The pipeline, run as a separate process so a crash or an out-of-memory
    condition can never take the interface with it.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
