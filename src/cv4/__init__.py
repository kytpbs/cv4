"""cv4 is a Python library for computer vision.""" # TODO: Update description

__version__ = "0.0.1"
try:
    import cv2  # pyright: ignore[reportUnusedImport]  # noqa: F401
except ImportError as e:
    raise ImportError(
        "cv4 requires OpenCV. Install it with `pip install cv4[opencv]`, or provide your own cv2 build."
    ) from e


def hello() -> str:
    return "Hello from cv4!"
