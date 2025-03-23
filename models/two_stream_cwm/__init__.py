from pathlib import Path
from typing import Annotated, Any, TypeVar

from torch import Tensor

T = TypeVar("T")

# helper types
TwoTuple = tuple[T, T]
ThreeTuple = tuple[T, T, T]
IntOr2Tuple = TwoTuple | int
Kwargs = dict[str, Any]
JSONDict = dict[str, Any]

PathLike = Path | str

ImageTensor = Annotated[Tensor, ["batch", "channel", "height", "width"]]
VideoTensor = Annotated[Tensor, ["batch", "channel", "time", "height", "width"]]


class VideoDims:
    BATCH = 0
    CHANNEL = 1
    TIME = 2
    HEIGHT = 3
    WIDTH = 4
