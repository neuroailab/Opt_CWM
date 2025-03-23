import json
from pathlib import Path
from typing import Callable, Type, TypeVar

import dacite

from models.two_stream_cwm import JSONDict, PathLike

T = TypeVar("T")
Caster = Callable[[JSONDict], object]


def load_from_json(target_class: Type[T], path: PathLike, caster: Caster | None = None) -> T:
    # coerce path to a proper pathlib.Path
    path = Path(path)
    assert path.is_file(), f"{path} not found"

    with path.open("r") as stream:
        json_dict = json.load(stream)
        if caster is not None:
            json_dict = caster(json_dict)
        return dacite.from_dict(data_class=target_class, data=json_dict)
