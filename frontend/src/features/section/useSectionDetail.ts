import { useCallback, useEffect, useState } from 'react';
import { api } from '../../services/api';
import type { Section } from '../../types/section';
import type { SectionInsights } from '../../types/insight';

export function useSectionDetail(sectionNumber: string) {
  const [section, setSection] = useState<Section | null>(null);
  const [sectionLoading, setSectionLoading] = useState(true);
  const [sectionError, setSectionError] = useState<string | null>(null);
  const [insights, setInsights] = useState<SectionInsights | null>(null);
  const [insightsLoading, setInsightsLoading] = useState(true);
  const [insightsError, setInsightsError] = useState<string | null>(null);

  const loadSection = useCallback(async () => {
    setSectionLoading(true); setSectionError(null);
    try { setSection(await api.section(sectionNumber)); }
    catch (error) { setSection(null); setSectionError(error instanceof Error ? error.message : 'Unable to load section.'); }
    finally { setSectionLoading(false); }
  }, [sectionNumber]);

  const loadInsights = useCallback(async () => {
    setInsightsLoading(true); setInsightsError(null);
    try { setInsights(await api.sectionInsights(sectionNumber)); }
    catch (error) { setInsights(null); setInsightsError(error instanceof Error ? error.message : 'Smart Insights unavailable.'); }
    finally { setInsightsLoading(false); }
  }, [sectionNumber]);

  useEffect(() => { void loadSection(); void loadInsights(); }, [loadSection, loadInsights]);

  return { section, sectionLoading, sectionError, retrySection: loadSection, insights, insightsLoading, insightsError, retryInsights: loadInsights };
}
