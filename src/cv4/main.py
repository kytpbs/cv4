from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np


@dataclass
class HSV:
    hue: float
    saturation: float
    value: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.hue, self.saturation, self.value)

    @property
    def np_array(self) -> np.ndarray[np.float64, np.dtype[np.float64]]:
        return np.array(self.as_tuple())


@dataclass
class RGB:
    red: float
    green: float
    blue: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.red, self.green, self.blue)

    def as_bgr(self) -> "BGR":
        return BGR(self.blue, self.green, self.red)


@dataclass
class Point:
    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def as_cv2_point(self) -> "cv2.typing.Point":
        return (int(self.x), int(self.y))


@dataclass
class BGR:
    blue: float
    green: float
    red: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.blue, self.green, self.red)

    def as_rgb(self) -> RGB:
        return RGB(self.red, self.green, self.blue)


@dataclass
class HSVRange:
    lower: HSV
    upper: HSV


class Image:
    def __init__(self, image: "cv2.typing.MatLike"):
        self.cv2_image: cv2.typing.MatLike = image

    def copy(self) -> "Image":
        return self.__class__(self.cv2_image.copy())


class DrawableImage(Image):
    def draw_contours(self, contours: list["Contour"], color: RGB, thickness: int = 2):
        _ = cv2.drawContours(
            self.cv2_image,
            [contour.cv2_image for contour in contours],
            -1,
            color.as_bgr().as_tuple(),
            thickness,
        )
        return self

    def draw_circle(self, center: Point, radius: int, color: RGB, thickness: int = -1):
        _ = cv2.circle(
            self.cv2_image,
            center.as_cv2_point(),
            radius,
            color.as_bgr().as_tuple(),
            thickness,
        )
        return self

    def draw_marker(  # noqa: PLR0913, PLR0917  # just replicating cv2 API
        self,
        position: Point,
        color: RGB,
        markerType: int = cv2.MARKER_CROSS,
        markerSize: int = 10,
        thickness: int = 2,
        lineType: int = cv2.LINE_AA,
    ):
        _ = cv2.drawMarker(
            self.cv2_image,
            position.as_cv2_point(),
            color.as_bgr().as_tuple(),
            markerType,
            markerSize,
            thickness,
            lineType,
        )
        return self


class BGRImage(DrawableImage):
    def __init__(self, image: "cv2.typing.MatLike"):
        super().__init__(image)

    @property
    def height(self) -> int:
        return self.cv2_image.shape[0]

    @property
    def width(self) -> int:
        return self.cv2_image.shape[1]

    def as_grayscale(self) -> "GrayscaleImage":
        return GrayscaleImage(cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2GRAY))

    def as_hsv(self) -> "HSVImage":
        return HSVImage(cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2HSV))

    def as_rgb(self) -> "RGBImage":
        return RGBImage(cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2RGB))


class HSVImage(Image):
    def __init__(self, image: "cv2.typing.MatLike"):
        super().__init__(image)

    def as_bgr(self) -> "BGRImage":
        return BGRImage(cv2.cvtColor(self.cv2_image, cv2.COLOR_HSV2BGR))

    def as_rgb(self) -> "RGBImage":
        return RGBImage(cv2.cvtColor(self.cv2_image, cv2.COLOR_HSV2RGB))

    def mask(self, range: HSVRange) -> "HSVMask":
        return HSVMask(
            cv2.inRange(self.cv2_image, range.lower.np_array, range.upper.np_array),
            range,
        )


class HSVMask(Image):
    def __init__(self, image: "cv2.typing.MatLike", range: HSVRange | None = None):
        super().__init__(image)

    def bitwise_or(self, other: "HSVMask") -> "HSVMask":
        self.cv2_image = cv2.bitwise_or(self.cv2_image, other.cv2_image)
        return self

    def clean(self, kernel_size: int) -> "HSVMask":
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.cv2_image = cv2.morphologyEx(self.cv2_image, cv2.MORPH_OPEN, kernel)
        self.cv2_image = cv2.morphologyEx(self.cv2_image, cv2.MORPH_CLOSE, kernel)
        return self

    def find_contours(self) -> "Contours":
        contours, _ = cv2.findContours(
            self.cv2_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return Contours([Contour(contour) for contour in contours])


class Contours(list["Contour"]):
    def __init__(self, contours: list["Contour"]):
        super().__init__(contours)

    def filter(self, function: Callable[["Contour"], bool]) -> "Contours":
        return Contours([contour for contour in self if function(contour)])


class Contour(Image):
    def __init__(self, image: "cv2.typing.MatLike"):
        super().__init__(image)
        self._moments: Final = cv2.moments(self.cv2_image)

    @property
    def area(self) -> float:
        return self._moments["m00"]

    @property
    def moments(self) -> "cv2.typing.Moments":
        return self._moments

    @property
    def cx(self) -> float:
        return self.centroid.x

    @property
    def cy(self) -> float:
        return self.centroid.y

    @property
    def centroid(self) -> Point:
        m00 = self._moments["m00"]
        if abs(m00) > 1e-9:  # noqa: PLR2004
            return Point(self._moments["m10"] / m00, self._moments["m01"] / m00)
        # degenerate contour (point / line) the mean of its vertices should be close enough.
        x, y = self.cv2_image.reshape(-1, 2).mean(axis=0)
        return Point(float(x), float(y))

    @property
    def base_y(self) -> float:
        return float(self.cv2_image[:, :, 1].max())  # pyright: ignore[reportAny]


class RGBImage(DrawableImage):
    def __init__(self, image: "cv2.typing.MatLike"):
        super().__init__(image)

    def as_bgr(self) -> "BGRImage":
        return BGRImage(cv2.cvtColor(self.cv2_image, cv2.COLOR_RGB2BGR))

    def as_hsv(self) -> "HSVImage":
        return HSVImage(cv2.cvtColor(self.cv2_image, cv2.COLOR_RGB2HSV))

    def as_grayscale(self) -> "GrayscaleImage":
        return GrayscaleImage(cv2.cvtColor(self.cv2_image, cv2.COLOR_RGB2GRAY))


class GrayscaleImage(Image):
    def __init__(self, image: "cv2.typing.MatLike"):
        super().__init__(image)

    def as_bgr(self) -> "BGRImage":
        return BGRImage(cv2.cvtColor(self.cv2_image, cv2.COLOR_GRAY2BGR))

    def as_rgb(self) -> "RGBImage":
        return RGBImage(cv2.cvtColor(self.cv2_image, cv2.COLOR_GRAY2RGB))
