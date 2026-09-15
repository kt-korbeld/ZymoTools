"""
Lightweight pipeline state, save as JSON.
The pipeline runs in distinct phases that may be separated by hours or days 
(the SLURM screening phase in particular). Hence progress is saved in a .json
"""

import json
from pathlib import Path

STATE_FILENAME = "pipeline.json"


class PipelineState:
    def __init__(self, data=None, path=None):
        self.data = dict(data or {})
        self.path = Path(path) if path else None

    # construction
    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.is_dir():
            path = path / STATE_FILENAME
        if not path.is_file():
            raise FileNotFoundError("No pipeline state at {}".format(path))
        return cls(json.loads(path.read_text()), path=path)

    @classmethod
    def load_or_empty(cls, path):
        try:
            return cls.load(path)
        except FileNotFoundError:
            target = Path(path)
            if target.is_dir():
                target = target / STATE_FILENAME
            return cls({}, path=target)

    # access
    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, **kwargs):
        # Store Paths as plain strings so the JSON stays portable.
        for k, v in kwargs.items():
            self.data[k] = str(v) if isinstance(v, Path) else v
        return self

    # save as json
    def save(self, path=None):
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("No path to save pipeline state to")
        if target.is_dir():
            target = target / STATE_FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.data, indent=4))
        self.path = target
        return target
