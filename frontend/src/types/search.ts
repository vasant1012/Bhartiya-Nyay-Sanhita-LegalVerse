export type SearchMode = 'recommended' | 'keyword' | 'semantic';

export type SectionSummary = {
  section_number: string;
  section_name: string;
  chapter_number: string;
  chapter_name: string;
};

export type MatchEvidence = {
  matched_terms: string[];
  concepts: string[];
  method: string;
};

export type SearchResult = {
  section: SectionSummary;
  excerpt: string;
  score: number;
  relevance: string;
  evidence: MatchEvidence;
};

export type SearchResponse = {
  query: string;
  mode_used: string;
  results: SearchResult[];
  partial_ml: boolean;
  warnings: string[];
};

export type SearchRequest = {
  query: string;
  mode: SearchMode;
  top_k: number;
  chapter_number?: string | null;
};
