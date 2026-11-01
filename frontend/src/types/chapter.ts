export type ChapterSummary = {
  chapter_number: string;
  chapter_name: string;
  section_count: number;
};

export type SectionSummary = {
  section_number: string;
  section_name: string;
  chapter_number: string;
  chapter_name: string;
};
