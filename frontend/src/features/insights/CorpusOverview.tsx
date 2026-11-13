import { useCallback, useEffect, useMemo, useState } from 'react';
import { BarChart3, BookOpen, CheckCircle2, CircleOff, Database, Network, RefreshCw, Sparkles } from 'lucide-react';
import { api, type HealthResponse } from '../../services/api';
import type { ChapterSummary, SectionSummary } from '../../types/chapter';

type CorpusState = {
  chapters: ChapterSummary[];
  sections: SectionSummary[];
  health: HealthResponse | null;
};

export function CorpusOverview() {
  const [data, setData] = useState<CorpusState>({ chapters: [], sections: [], health: null });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const [chapters, sections, health] = await Promise.all([
        api.chapters(),
        api.sections(),
        api.health(),
      ]);
      setData({ chapters, sections, health });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to load corpus overview.');
    } finally { setLoading(false); }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const maxSections = useMemo(() => Math.max(1, ...data.chapters.map(chapter => chapter.section_count)), [data.chapters]);
  const coverage = useMemo(() => data.chapters.reduce((sum, chapter) => sum + chapter.section_count, 0), [data.chapters]);

  if (loading) return <div className="p-7"><div className="space-y-5"><div className="grid grid-cols-3 gap-4">{[1,2,3].map(item => <div key={item} className="h-28 animate-pulse rounded-2xl bg-white" />)}</div><div className="h-[430px] animate-pulse rounded-2xl bg-white" /></div></div>;

  if (error) return <div className="p-7"><div className="rounded-2xl border border-red-200 bg-red-50 p-7"><h2 className="text-xl font-bold text-red-800">Corpus overview could not be loaded</h2><p className="mt-2 text-sm text-red-700">{error}</p><button onClick={() => void load()} className="mt-5 flex items-center gap-2 rounded-lg bg-white px-4 py-2 text-sm font-semibold text-red-700 ring-1 ring-red-200"><RefreshCw size={15}/>Try again</button></div></div>;

  return <div className="mx-auto max-w-7xl space-y-5 p-7">
    <section className="flex items-start justify-between rounded-2xl border border-legal-border bg-white p-6 shadow-sm"><div><div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-legal-purple"><Sparkles size={15}/>BNS Intelligence</div><h2 className="mt-2 text-2xl font-bold text-legal-text">Corpus Overview</h2><p className="mt-2 max-w-3xl text-sm leading-6 text-legal-muted">A source-backed structural view of the BNS corpus. Intelligence features are shown only when their backend capability is available.</p></div><div className="rounded-xl bg-violet-50 p-3 text-legal-purple"><BarChart3 size={25}/></div></section>

    <section className="grid grid-cols-3 gap-4">
      <Metric icon={<Database size={20}/>} label="Sections" value={data.sections.length} note="Loaded from the source API" />
      <Metric icon={<BookOpen size={20}/>} label="Chapters" value={data.chapters.length} note="Distinct chapter records" />
      <Metric icon={<Network size={20}/>} label="Chapter coverage" value={coverage} note="Sections represented across chapters" />
    </section>

    <section className="grid grid-cols-[minmax(0,1fr)_330px] gap-5">
      <div className="rounded-2xl border border-legal-border bg-white p-6 shadow-sm">
        <div><h3 className="font-bold text-legal-text">Sections by chapter</h3><p className="mt-1 text-xs text-legal-muted">Structural distribution from the Chapters API.</p></div>
        <div className="mt-6 max-h-[440px] space-y-3 overflow-y-auto pr-2">{data.chapters.map(chapter => {
          const width=Math.max(2,(chapter.section_count/maxSections)*100);
          return <div key={`${chapter.chapter_number}-${chapter.chapter_name}`} className="grid grid-cols-[44px_minmax(180px,260px)_1fr_38px] items-center gap-3"><div className="text-xs font-bold text-legal-gold">{chapter.chapter_number}</div><div className="truncate text-xs font-medium text-legal-text" title={chapter.chapter_name}>{chapter.chapter_name}</div><div className="h-2.5 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full bg-legal-navy" style={{width:`${width}%`}} /></div><div className="text-right text-xs font-semibold text-legal-muted">{chapter.section_count}</div></div>;
        })}</div>
      </div>

      <div className="space-y-5">
        <CapabilityCard title="Keyword Search" available={Boolean(data.health?.availability.keyword_search)} copy="Lexical search capability" />
        <CapabilityCard title="Semantic Search" available={Boolean(data.health?.availability.semantic_search)} copy="FAISS-backed semantic retrieval" />
        <CapabilityCard title="Graph Insights" available={Boolean(data.health?.availability.graph_insights)} copy="Neo4j connectivity and graph capability" />
        <div className="rounded-2xl border border-violet-100 bg-violet-50/60 p-5"><div className="text-xs font-bold uppercase tracking-wider text-legal-purple">Coming through graph enrichment</div><div className="mt-3 space-y-2 text-xs leading-5 text-legal-muted"><div>Concept Explorer</div><div>Section Relationships</div><div>Chapter Intelligence</div></div><p className="mt-4 text-[11px] leading-5 text-slate-500">These views should populate only after corresponding graph relationships are generated and validated.</p></div>
      </div>
    </section>
  </div>;
}

function Metric({icon,label,value,note}:{icon:React.ReactNode;label:string;value:number;note:string}){return <div className="rounded-2xl border border-legal-border bg-white p-5 shadow-sm"><div className="flex items-start justify-between"><div><div className="text-xs font-semibold uppercase tracking-wider text-legal-muted">{label}</div><div className="mt-2 text-3xl font-bold text-legal-text">{value}</div><div className="mt-2 text-xs text-legal-muted">{note}</div></div><div className="rounded-xl bg-slate-100 p-2.5 text-legal-navy">{icon}</div></div></div>}
function CapabilityCard({title,available,copy}:{title:string;available:boolean;copy:string}){return <div className="rounded-2xl border border-legal-border bg-white p-4 shadow-sm"><div className="flex items-start gap-3">{available?<CheckCircle2 size={19} className="mt-0.5 shrink-0 text-emerald-600"/>:<CircleOff size={19} className="mt-0.5 shrink-0 text-slate-400"/>}<div><div className="text-sm font-semibold text-legal-text">{title}</div><div className="mt-1 text-xs text-legal-muted">{copy}</div><div className={`mt-2 text-[10px] font-bold uppercase tracking-wider ${available?'text-emerald-700':'text-slate-500'}`}>{available?'Available':'Unavailable'}</div></div></div></div>}
