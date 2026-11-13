import { useEffect, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import { api, type HealthResponse } from '../services/api';

export function Header({ title, subtitle }: { title: string; subtitle: string }) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(false);
  const check = async () => { setLoading(true); setError(false); try { setHealth(await api.health()); } catch { setHealth(null); setError(true); } finally { setLoading(false); } };
  useEffect(() => { void check(); }, []);
  const connected = Boolean(health?.availability.source_data);
  return (
    <header className="flex h-[74px] items-center justify-between border-b border-legal-border bg-white px-7">
      <div><h1 className="text-xl font-bold text-legal-text">{title}</h1><p className="mt-0.5 text-xs text-legal-muted">{subtitle}</p></div>
      <button onClick={() => void check()} className="flex items-center gap-2 rounded-lg border border-legal-border bg-white px-3 py-2 text-xs text-legal-muted shadow-sm hover:bg-slate-50">
        <span className={`size-2 rounded-full ${connected ? 'bg-emerald-500' : error ? 'bg-amber-500' : 'bg-slate-300'}`} />
        {connected ? 'API connected' : error ? 'API unavailable' : 'Checking API'}
        <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
      </button>
    </header>
  );
}
