import { FormEvent, useCallback, useEffect, useState } from 'react';
import {
  AlertTriangle, ArrowRight, BookOpen, ChevronDown, Info, Search,
  ShieldCheck, Sparkles, Tags, X,
} from 'lucide-react';
import { api } from '../services/api';
import type { SearchMode, SearchResponse, SearchResult } from '../types/search';

type Props = {
  initialQuery: string;
  onOpenSection: (sectionNumber: string) => void;
};

const relevanceStyle: Record<string, string> = {
  very_strong: 'bg-emerald-50 text-emerald-700 ring-emerald-200',
  strong: 'bg-blue-50 text-blue-700 ring-blue-200',
  possible: 'bg-amber-50 text-amber-700 ring-amber-200',
  related: 'bg-slate-100 text-slate-600 ring-slate-200',
};

const relevanceLabel = (value: string) =>
  ({ very_strong: 'Very strong match', strong: 'Strong match', possible: 'Possible match', related: 'Related' }[value] ?? value.replaceAll('_', ' '));

function MatchDrawer({ result, onClose }: { result: SearchResult; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-slate-950/25" onMouseDown={onClose}>
      <aside className="h-full w-[430px] overflow-y-auto bg-white p-6 shadow-2xl" onMouseDown={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between gap-4 border-b border-legal-border pb-5">
          <div><div className="text-xs font-bold uppercase tracking-wider text-legal-purple">Smart Insight</div><h3 className="mt-1 text-xl font-bold text-legal-text">Why this section appeared</h3></div>
          <button onClick={onClose} className="rounded-lg p-2 text-slate-500 hover:bg-slate-100" aria-label="Close explanation"><X size={18}/></button>
        </div>
        <div className="mt-6 rounded-xl bg-slate-50 p-4"><div className="text-xs font-semibold text-legal-muted">SECTION</div><div className="mt-1 font-bold text-legal-text">{result.section.section_number} · {result.section.section_name}</div></div>
        <div className="mt-6 space-y-5">
          <div><div className="flex items-center gap-2 font-semibold text-legal-text"><ShieldCheck size={18} className="text-legal-purple"/>Search method</div><p className="mt-2 text-sm leading-6 text-legal-muted">{result.evidence.method || 'Search relevance'}</p></div>
          <div><div className="flex items-center gap-2 font-semibold text-legal-text"><Tags size={18} className="text-legal-purple"/>Matched terms</div>{result.evidence.matched_terms?.length ? <div className="mt-3 flex flex-wrap gap-2">{result.evidence.matched_terms.map(term => <span key={term} className="rounded-full bg-violet-50 px-2.5 py-1 text-xs font-medium text-legal-purple">{term}</span>)}</div> : <p className="mt-2 text-sm text-legal-muted">No direct lexical terms were supplied for this result.</p>}</div>
          {result.evidence.concepts?.length > 0 && <div><div className="font-semibold text-legal-text">Related concepts</div><div className="mt-3 flex flex-wrap gap-2">{result.evidence.concepts.map(concept => <span key={concept} className="rounded-full bg-violet-50 px-2.5 py-1 text-xs text-legal-purple">{concept}</span>)}</div></div>}
          <div className="rounded-xl border border-violet-100 bg-violet-50/60 p-4 text-sm leading-6 text-slate-600"><Info size={16} className="mb-2 text-legal-purple"/>The ranking is a research aid. Open the section and read the original BNS provision before drawing conclusions.</div>
        </div>
      </aside>
    </div>
  );
}

export function SmartSearch({ initialQuery, onOpenSection }: Props) {
  const [query, setQuery] = useState(initialQuery);
  const [mode, setMode] = useState<SearchMode>('recommended');
  const [data, setData] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [explaining, setExplaining] = useState<SearchResult | null>(null);

  const execute = useCallback(async (text: string, searchMode: SearchMode) => {
    const clean = text.trim(); if (!clean) return;
    setLoading(true); setError(null);
    try { setData(await api.search({ query: clean, mode: searchMode, top_k: 8, chapter_number: null })); }
    catch (e) { setData(null); setError(e instanceof Error ? e.message : 'Search failed'); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { if (initialQuery.trim()) void execute(initialQuery, 'recommended'); }, [initialQuery, execute]);

  const submit = (event: FormEvent) => { event.preventDefault(); void execute(query, mode); };
  const changeMode = (next: SearchMode) => { setMode(next); if (query.trim()) void execute(query, next); };

  return (
    <div className="mx-auto max-w-7xl p-7">
      <form onSubmit={submit} className="rounded-2xl border border-legal-border bg-white p-5 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="flex flex-1 items-center rounded-xl border border-slate-300 bg-white px-4 focus-within:border-legal-navy focus-within:ring-2 focus-within:ring-blue-100"><Search size={19} className="mr-3 text-slate-400"/><input value={query} onChange={(e)=>setQuery(e.target.value)} className="w-full py-3.5 text-sm text-legal-text outline-none placeholder:text-slate-400" placeholder="Search section number, legal concept, or describe a situation..." /></div>
          <div className="relative"><select value={mode} onChange={(e)=>changeMode(e.target.value as SearchMode)} className="h-12 appearance-none rounded-xl border border-slate-300 bg-white pl-4 pr-10 text-sm font-medium text-legal-text outline-none"><option value="recommended">Recommended</option><option value="keyword">Keyword</option><option value="semantic">Semantic</option></select><ChevronDown size={15} className="pointer-events-none absolute right-3 top-4 text-slate-400"/></div>
          <button disabled={!query.trim() || loading} className="h-12 rounded-xl bg-legal-navy px-5 text-sm font-bold text-white disabled:opacity-50">{loading ? 'Searching…' : 'Search'}</button>
        </div>
      </form>

      {data?.partial_ml && <div className="mt-4 flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800"><AlertTriangle size={18} className="mt-0.5 shrink-0"/><div><strong>Smart search partially available.</strong>{data.warnings?.length > 0 && <span> {data.warnings.join(' ')}</span>}</div></div>}
      {error && <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700"><strong>Search could not be completed.</strong><div className="mt-1">{error}</div><button onClick={()=>void execute(query,mode)} className="mt-3 rounded-lg bg-white px-3 py-1.5 font-semibold ring-1 ring-red-200">Try again</button></div>}

      {loading && <div className="mt-5 space-y-3">{[1,2,3].map(i=><div key={i} className="h-40 animate-pulse rounded-2xl border border-slate-200 bg-white p-5"><div className="h-4 w-24 rounded bg-slate-100"/><div className="mt-3 h-5 w-72 rounded bg-slate-100"/><div className="mt-5 h-3 w-full rounded bg-slate-100"/><div className="mt-2 h-3 w-4/5 rounded bg-slate-100"/></div>)}</div>}

      {!loading && data && <section className="mt-6">
        <div className="mb-4 flex items-end justify-between"><div><h2 className="text-xl font-bold text-legal-text">Relevant BNS sections</h2><p className="mt-1 text-sm text-legal-muted">Ranked results for “{data.query}” · {data.mode_used} search</p></div><div className="text-xs font-semibold text-legal-muted">{data.results.length} result{data.results.length===1?'':'s'}</div></div>
        {data.results.length === 0 ? <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-10 text-center"><Search className="mx-auto text-slate-400"/><h3 className="mt-4 font-bold text-legal-text">No matching sections found</h3><p className="mt-2 text-sm text-legal-muted">Try changing the wording or switch the search strategy.</p></div> : <div className="space-y-4">{data.results.map((result) => <article key={result.section.section_number} className="rounded-2xl border border-legal-border bg-white p-5 shadow-sm transition hover:shadow-md">
          <div className="flex items-start justify-between gap-5"><div><div className="text-xs font-bold uppercase tracking-wider text-legal-gold">Section {result.section.section_number}</div><h3 className="mt-1 text-lg font-bold text-legal-text">{result.section.section_name}</h3><div className="mt-1 flex items-center gap-2 text-xs text-legal-muted"><BookOpen size={14}/>Chapter {result.section.chapter_number} · {result.section.chapter_name}</div></div><span className={`rounded-full px-3 py-1 text-xs font-bold capitalize ring-1 ${relevanceStyle[result.relevance] ?? relevanceStyle.related}`}>{relevanceLabel(result.relevance)}</span></div>
          <p className="mt-4 text-sm leading-6 text-slate-600">{result.excerpt}</p>
          {result.evidence.matched_terms?.length > 0 && <div className="mt-4 flex flex-wrap gap-2">{result.evidence.matched_terms.slice(0,6).map(term=><span key={term} className="rounded-full bg-slate-100 px-2.5 py-1 text-xs text-slate-600">{term}</span>)}</div>}
          <div className="mt-5 flex items-center gap-2 border-t border-slate-100 pt-4"><button onClick={()=>onOpenSection(result.section.section_number)} className="flex items-center gap-2 rounded-lg bg-legal-navy px-3.5 py-2 text-xs font-bold text-white">Read section <ArrowRight size={14}/></button><button onClick={()=>setExplaining(result)} className="flex items-center gap-2 rounded-lg border border-slate-200 px-3.5 py-2 text-xs font-semibold text-legal-purple hover:bg-violet-50"><Sparkles size={14}/>Why this matched</button></div>
        </article>)}</div>}
      </section>}

      {!loading && !data && !error && <div className="mt-8 text-center text-sm text-legal-muted">Enter a section, legal concept, or plain-language situation to begin.</div>}
      {explaining && <MatchDrawer result={explaining} onClose={()=>setExplaining(null)} />}
    </div>
  );
}
