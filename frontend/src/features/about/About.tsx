import { BookOpen, BrainCircuit, Database, GitBranch, Scale, Search, ShieldCheck } from 'lucide-react';

export function About() {
  return <div className="mx-auto max-w-6xl space-y-5 p-7">
    <section className="rounded-3xl bg-gradient-to-br from-legal-navy to-legal-deep p-8 text-white shadow-lg"><div className="flex items-start gap-5"><div className="grid size-14 shrink-0 place-items-center rounded-2xl bg-legal-gold"><Scale size={28}/></div><div><div className="text-xs font-bold uppercase tracking-wider text-amber-200">LegalVerse</div><h2 className="mt-2 text-3xl font-bold">Bharatiya Nyaya Sanhita Intelligence Explorer</h2><p className="mt-3 max-w-3xl text-sm leading-6 text-slate-200">A source-first research demo that combines statutory browsing with lexical search, semantic retrieval and graph-based relationships.</p></div></div></section>

    <section className="grid grid-cols-3 gap-4">
      <InfoCard icon={<BookOpen/>} title="Source First" copy="Original BNS section text remains the authoritative content displayed by the application." />
      <InfoCard icon={<Search/>} title="Hybrid Retrieval" copy="Keyword and FAISS semantic retrieval support the Smart Search experience." />
      <InfoCard icon={<GitBranch/>} title="Graph Intelligence" copy="Neo4j stores validated relationships used by Smart Insights and relationship exploration." />
    </section>

    <section className="grid grid-cols-[1fr_360px] gap-5">
      <div className="rounded-2xl border border-legal-border bg-white p-6 shadow-sm"><h3 className="font-bold text-legal-text">How LegalVerse separates information</h3><div className="mt-5 space-y-4"><Layer color="bg-amber-500" title="Statutory source" copy="Section numbers, titles, chapters and original descriptions from the BNS dataset."/><Layer color="bg-slate-400" title="Deterministic metadata" copy="Structural information and explicit references where available."/><Layer color="bg-violet-500" title="ML-derived intelligence" copy="Semantic search, related provisions and future validated concept relationships."/></div></div>
      <div className="space-y-4"><div className="rounded-2xl border border-violet-100 bg-violet-50 p-5"><div className="flex items-center gap-2 font-semibold text-legal-purple"><BrainCircuit size={18}/>Intelligence stack</div><div className="mt-4 space-y-2 text-xs text-legal-muted"><div className="flex items-center gap-2"><Database size={14}/>FAISS vector retrieval</div><div className="flex items-center gap-2"><GitBranch size={14}/>Neo4j knowledge relationships</div><div className="flex items-center gap-2"><Search size={14}/>Lexical + hybrid search</div></div></div><div className="rounded-2xl border border-amber-200 bg-amber-50 p-5"><div className="flex items-center gap-2 font-semibold text-amber-800"><ShieldCheck size={18}/>Research aid</div><p className="mt-3 text-xs leading-5 text-amber-800">LegalVerse is intended for research and educational demonstration. Search rankings and Smart Insights are not legal advice and do not establish that a provision applies to a particular matter.</p></div></div>
    </section>
  </div>;
}

function InfoCard({icon,title,copy}:{icon:React.ReactNode;title:string;copy:string}){return <div className="rounded-2xl border border-legal-border bg-white p-5 shadow-sm"><div className="grid size-10 place-items-center rounded-xl bg-slate-100 text-legal-navy">{icon}</div><h3 className="mt-4 font-bold text-legal-text">{title}</h3><p className="mt-2 text-sm leading-6 text-legal-muted">{copy}</p></div>}
function Layer({color,title,copy}:{color:string;title:string;copy:string}){return <div className="flex gap-3"><div className={`mt-1.5 size-2.5 shrink-0 rounded-full ${color}`}/><div><div className="text-sm font-semibold text-legal-text">{title}</div><p className="mt-1 text-xs leading-5 text-legal-muted">{copy}</p></div></div>}
