import { useCallback, useEffect, useState } from 'react';
import { api } from '../../services/api';
import type { ChapterSummary, SectionSummary } from '../../types/chapter';

export function useChapters() {
  const [chapters, setChapters] = useState<ChapterSummary[]>([]);
  const [selectedChapter, setSelectedChapter] = useState<string>('');
  const [sections, setSections] = useState<SectionSummary[]>([]);
  const [chaptersLoading, setChaptersLoading] = useState(true);
  const [sectionsLoading, setSectionsLoading] = useState(false);
  const [chaptersError, setChaptersError] = useState<string | null>(null);
  const [sectionsError, setSectionsError] = useState<string | null>(null);

  const loadChapters = useCallback(async () => {
    setChaptersLoading(true); setChaptersError(null);
    try {
      const data = await api.chapters();
      setChapters(data);
      setSelectedChapter(current => current || data[0]?.chapter_number || '');
    } catch (error) {
      setChapters([]);
      setChaptersError(error instanceof Error ? error.message : 'Unable to load chapters.');
    } finally { setChaptersLoading(false); }
  }, []);

  const loadSections = useCallback(async (chapterNumber: string) => {
    if (!chapterNumber) { setSections([]); return; }
    setSectionsLoading(true); setSectionsError(null);
    try { setSections(await api.sectionsByChapter(chapterNumber)); }
    catch (error) { setSections([]); setSectionsError(error instanceof Error ? error.message : 'Unable to load sections.'); }
    finally { setSectionsLoading(false); }
  }, []);

  useEffect(() => { void loadChapters(); }, [loadChapters]);
  useEffect(() => { if (selectedChapter) void loadSections(selectedChapter); }, [selectedChapter, loadSections]);

  return { chapters, selectedChapter, setSelectedChapter, sections, chaptersLoading, sectionsLoading, chaptersError, sectionsError, retryChapters: loadChapters, retrySections: () => loadSections(selectedChapter) };
}
