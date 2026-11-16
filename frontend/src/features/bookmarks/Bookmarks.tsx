import { useMemo, useState } from 'react';
import { ArrowRight, Bookmark as BookmarkIcon, Search, Trash2 } from 'lucide-react';
import type { Bookmark } from '../../types/bookmark';

export function Bookmarks({ bookmarks, onOpenSection, onRemove }: { bookmarks: Bookmark[]; onOpenSection: (sectionNumber: string) => void; onRemove: (sectionNumber: string) => void }) {
  const [query, setQuery] = useState('');
  const filtered = useMemo(() => {
    const clean = query.trim().toLowerCase();
    if (!clean) return bookmarks;
    return bookmarks.filter(item =>
      item.section_number.toLowerCase().includes(clean) ||
      item.section_name.toLowerCase().includes(clean) ||
      item.chapter_name.toLowerCase().includes(clean)
    );
  }, [bookmarks, query]);

  return <div className="mx-auto max-w-6xl p-7">
    <section className="rounded-2xl border border-legal-border bg-white p-6 shadow-sm">
      <div className="flex items-start justify-between gap-5"><div><div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-legal-gold"><BookmarkIcon size={15}/>Saved Research</div><h2 className="mt-2 text-2xl font-bold text-legal-text">Bookmarks</h2><p className="mt-2 text-sm text-legal-muted">Saved locally in this browser for the LegalVerse demo.</p></div><div className="rounded-xl bg-amber-50 px-4 py-3 text-center"><div className="text-2xl font-bold text-legal-text">{bookmarks.length}</div><div className="text-[10px] font-bold uppercase tracking-wider text-amber-700">Saved</div></div></div>
      <div className="mt-6 flex items-center rounded-xl border border-slate-300 px-3"><Search size={17} className="mr-2 text-slate-400"/><input value={query} onChange={event=>setQuery(event.target.value)} className="w-full py-3 text-sm outline-none" placeholder="Search saved sections..."/></div>
    </section>

    <section className="mt-5">
      {!bookmarks.length ? <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-12 text-center"><BookmarkIcon className="mx-auto text-slate-400"/><h3 className="mt-4 font-bold text-legal-text">No bookmarked sections yet</h3><p className="mt-2 text-sm text-legal-muted">Save a section from Section Detail to keep it here.</p></div> : !filtered.length ? <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-10 text-center text-sm text-legal-muted">No saved sections match “{query}”.</div> : <div className="grid grid-cols-2 gap-4">{filtered.map(item => <article key={item.section_number} className="rounded-2xl border border-legal-border bg-white p-5 shadow-sm"><div className="text-xs font-bold uppercase tracking-wider text-legal-gold">Section {item.section_number}</div><h3 className="mt-2 font-bold text-legal-text">{item.section_name}</h3><p className="mt-2 text-xs leading-5 text-legal-muted">Chapter {item.chapter_number} · {item.chapter_name}</p><div className="mt-5 flex items-center justify-between border-t border-slate-100 pt-4"><button onClick={()=>onOpenSection(item.section_number)} className="flex items-center gap-2 rounded-lg bg-legal-navy px-3 py-2 text-xs font-bold text-white">Read section <ArrowRight size={13}/></button><button onClick={()=>onRemove(item.section_number)} className="rounded-lg p-2 text-slate-400 hover:bg-red-50 hover:text-red-600" aria-label={`Remove bookmark for section ${item.section_number}`}><Trash2 size={16}/></button></div></article>)}</div>}
    </section>
  </div>;
}
