import { Search } from 'lucide-react';
export function SearchPlaceholder({initialQuery}:{initialQuery:string}){
  return <div className="p-7"><div className="rounded-2xl border border-legal-border bg-white p-7 shadow-sm"><div className="flex items-center gap-3"><div className="grid size-11 place-items-center rounded-xl bg-blue-50 text-legal-navy"><Search size={21}/></div><div><h2 className="font-bold text-legal-text">Smart Search handoff</h2><p className="text-sm text-legal-muted">Search execution will be implemented in Phase 4.3.</p></div></div>{initialQuery&&<div className="mt-5 rounded-xl bg-slate-50 px-4 py-3 text-sm"><span className="font-semibold text-legal-text">Query preserved:</span> <span className="text-legal-muted">{initialQuery}</span></div>}</div></div>
}
