import { FormEvent, useEffect, useState } from "react";
import {
  ArrowRight,
  BookOpen,
  Network,
  Search,
  Sparkles,
  CheckCircle2,
  CircleOff,
} from "lucide-react";
import { api, type HealthResponse } from "../services/api";
import type { Navigate } from "../types/app";

const examples = [
  "criminal breach of trust",
  "property taken dishonestly",
  "public servant",
];

export function Explore({ navigate }: { navigate: Navigate }) {
  const [query, setQuery] = useState("");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch(() => setHealthError(true));
  }, []);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const clean = query.trim();
    if (clean) navigate("search", { query: clean });
  };

  return (
    <div className="mx-auto max-w-7xl space-y-6 p-7">
      <section className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-legal-navy to-legal-deep p-9 text-white shadow-lg">
        <div className="absolute -right-20 -top-28 size-80 rounded-full bg-white/5" />
        <div className="relative max-w-4xl">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/10 px-3 py-1.5 text-xs font-semibold text-amber-200">
            <Sparkles size={14} />
            BNS LEGAL INTELLIGENCE
          </div>
          <h2 className="text-4xl font-bold leading-tight">
            Explore the Bharatiya Nyaya Sanhita with intelligent search.
          </h2>
          <p className="mt-4 max-w-3xl text-base leading-7 text-slate-200">
            Find statutory provisions using section numbers, legal terminology,
            or plain-language descriptions, then inspect the original BNS source
            and related insights.
          </p>
          <form
            onSubmit={submit}
            className="mt-7 flex overflow-hidden rounded-xl bg-white p-1.5 shadow-xl"
          >
            <div className="flex flex-1 items-center px-3">
              <Search size={19} className="mr-3 text-slate-400" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="w-full border-0 bg-transparent py-3 text-sm text-slate-800 outline-none placeholder:text-slate-400"
                placeholder="Search a section or describe a situation..."
                aria-label="Search BNS"
              />
            </div>
            <button
              disabled={!query.trim()}
              className="flex items-center gap-2 rounded-lg bg-legal-gold px-5 py-3 text-sm font-bold text-white enabled:hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Smart Search <ArrowRight size={17} />
            </button>
          </form>
          <div className="mt-3 flex items-center gap-2 text-xs text-slate-300">
            <span>Try:</span>
            {examples.map((example) => (
              <button
                key={example}
                onClick={() => setQuery(example)}
                className="rounded-full border border-white/15 bg-white/5 px-2.5 py-1 hover:bg-white/10"
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      </section>

      <div className="grid grid-cols-3 gap-5">
        {[
          {
            title: "Smart Search",
            copy: "Find relevant BNS provisions using ordinary language.",
            icon: Search,
            page: "search" as const,
            color: "bg-blue-50 text-legal-navy",
          },
          {
            title: "Browse Chapters",
            copy: "Navigate the BNS structurally by chapter and section.",
            icon: BookOpen,
            page: "chapters" as const,
            color: "bg-amber-50 text-amber-700",
          },
          {
            title: "Smart Insights",
            copy: "Explore concepts and relationships across provisions.",
            icon: Network,
            page: "insights" as const,
            color: "bg-violet-50 text-legal-purple",
          },
        ].map(({ title, copy, icon: Icon, page, color }) => (
          <button
            key={title}
            onClick={() => navigate(page)}
            className="group rounded-2xl border border-legal-border bg-white p-6 text-left shadow-sm transition hover:-translate-y-0.5 hover:shadow-md"
          >
            <div
              className={`grid size-11 place-items-center rounded-xl ${color}`}
            >
              <Icon size={22} />
            </div>
            <h3 className="mt-5 font-bold text-legal-text">{title}</h3>
            <p className="mt-2 text-sm leading-6 text-legal-muted">{copy}</p>
            <div className="mt-5 flex items-center gap-1 text-xs font-bold text-legal-navy">
              Open <ArrowRight size={14} />
            </div>
          </button>
        ))}
      </div>

      <section className="rounded-2xl border border-legal-border bg-white p-5 shadow-sm">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-bold text-legal-text">
              Intelligence availability
            </h3>
            <p className="mt-1 text-xs text-legal-muted">
              Live status from the LegalVerse backend.
            </p>
          </div>
          {healthError && (
            <span className="text-xs font-semibold text-amber-700">
              Backend unavailable
            </span>
          )}
        </div>
        {health ? (
          <div className="mt-4 grid grid-cols-4 gap-3">
            {[
              ["Source data", health.availability.source_data],
              ["Keyword search", health.availability.keyword_search],
              ["Semantic search", health.availability.semantic_search],
              ["Graph insights", health.availability.graph_insights],
            ].map(([label, available]) => (
              <div
                key={String(label)}
                className="flex items-center gap-2 rounded-xl bg-slate-50 px-3 py-3 text-sm"
              >
                {available ? (
                  <CheckCircle2 size={17} className="text-emerald-600" />
                ) : (
                  <CircleOff size={17} className="text-slate-400" />
                )}
                <span className="text-legal-text">{String(label)}</span>
              </div>
            ))}
          </div>
        ) : (
          !healthError && (
            <div className="mt-4 h-11 animate-pulse rounded-xl bg-slate-100" />
          )
        )}
      </section>
    </div>
  );
}
