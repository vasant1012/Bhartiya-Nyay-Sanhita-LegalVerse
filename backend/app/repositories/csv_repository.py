from __future__ import annotations
from pathlib import Path
import pandas as pd
from app.schemas.section import ChapterSummary, Section

ALIASES = {
    'chapter_number': ['Chapter Number', 'Chapter_Number', 'chapter_number'],
    'chapter_name': ['Chapter Name', 'Chapter_Name', 'chapter_name'],
    'chapter_subtype': ['Chapter Subtype', 'Chapter_Subtype', 'chapter_subtype'],
    'section_number': ['Section Number', 'Section_Number', 'section_number'],
    'section_name': ['Section Name', 'Section_Name', 'section_name'],
    'description': ['Description', 'description'],
}

class CsvSectionRepository:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._frame: pd.DataFrame | None = None

    def load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f'BNS data file not found: {self.path}')
        frame = pd.read_csv(self.path, dtype=str).fillna('')
        rename = {}
        for target, candidates in ALIASES.items():
            source = next((c for c in candidates if c in frame.columns), None)
            if source is None and target != 'chapter_subtype':
                raise ValueError(f'Missing required CSV column for {target}')
            if source:
                rename[source] = target
        self._frame = frame.rename(columns=rename)
        if 'chapter_subtype' not in self._frame:
            self._frame['chapter_subtype'] = ''
        self._frame['section_number'] = self._frame['section_number'].astype(str).str.strip()

    @property
    def frame(self) -> pd.DataFrame:
        if self._frame is None:
            self.load()
        return self._frame.copy()

    def list_sections(self, chapter_number: str | None = None) -> list[Section]:
        frame = self.frame
        if chapter_number is not None:
            frame = frame[frame.chapter_number.astype(str) == str(chapter_number)]
        return [Section(**row) for row in frame.to_dict('records')]

    def get_section(self, section_number: str) -> Section | None:
        frame = self.frame
        match = frame[frame.section_number.astype(str) == str(section_number).strip()]
        return None if match.empty else Section(**match.iloc[0].to_dict())

    def list_chapters(self) -> list[ChapterSummary]:
        grouped = (self.frame.groupby(['chapter_number', 'chapter_name'], sort=False)
                   .size().reset_index(name='section_count'))
        return [ChapterSummary(**row) for row in grouped.to_dict('records')]
