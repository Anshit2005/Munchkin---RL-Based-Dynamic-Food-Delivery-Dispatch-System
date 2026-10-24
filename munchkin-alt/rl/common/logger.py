from __future__ import annotations

import csv
from pathlib import Path


class CSVLogger:
    """Append-only CSV logger; the header is fixed by the first row."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = None

    def log(self, row: dict):
        if self._w is None:
            self._w = csv.DictWriter(self._f, fieldnames=list(row))
            self._w.writeheader()
        self._w.writerow(row)
        self._f.flush()

    def close(self):
        self._f.close()
