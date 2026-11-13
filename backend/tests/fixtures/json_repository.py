import json
from pathlib import Path

import pandas as pd

from backend.repositories.csv_repository import (
    ALIASES,
    CsvSectionRepository,
)


class JsonFixtureRepository(CsvSectionRepository):

    def __init__(self, path: Path):
        self.path = Path(path)
        self._frame = None

    def load(self) -> None:

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as file:

            records = json.load(file)

        frame = pd.DataFrame(records)

        rename = {}

        for target, candidates in ALIASES.items():

            source = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate in frame.columns
                ),
                None,
            )

            if source is None and target != "chapter_subtype":

                raise ValueError(
                    f"Missing required fixture field: {target}"
                )

            if source:
                rename[source] = target

        frame = frame.rename(columns=rename)

        if "chapter_subtype" not in frame.columns:
            frame["chapter_subtype"] = ""

        frame = frame.fillna("")

        frame["section_number"] = (
            frame["section_number"]
            .astype(str)
            .str.strip()
        )

        self._frame = frame