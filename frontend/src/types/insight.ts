import type {
  SectionSummary,
} from './search';


export type RelatedSection = {
  section: SectionSummary;

  score: number | null;

  relationship: string;

  explanation: string | null;
};


export type SectionInsights = {
  section_number: string;

  concepts: string[];

  cross_references: string[];

  related_sections: RelatedSection[];

  partial_ml: boolean;

  warnings: string[];
};