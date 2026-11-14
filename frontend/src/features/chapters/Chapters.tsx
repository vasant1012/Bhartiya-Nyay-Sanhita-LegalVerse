import { useMemo, useState } from "react";
import {
  ArrowRight,
  BookOpen,
  FileText,
  RefreshCw,
  Search,
} from "lucide-react";
import { useChapters } from "./useChapters";

export function Chapters({
  onOpenSection,
}: {
  onOpenSection: (sectionNumber: string) => void;
}) {
  const state = useChapters();
  const [filter, setFilter] = useState("");
  const selected =
    state.chapters.find((c) => c.chapter_number === state.selectedChapter) ??
    null;
  const filtered = useMemo(() => {
    const query = filter.trim().toLowerCase();
    if (!query) return state.chapters;
    return state.chapters.filter(
      (chapter) =>
        chapter.chapter_number.toLowerCase().includes(query) ||
        chapter.chapter_name.toLowerCase().includes(query),
    );
  }, [state.chapters, filter]);

  if (state.chaptersLoading)
    return (
      <div className="p-7">
        <div className="grid min-h-[600px] grid-cols-[330px_1fr] overflow-hidden rounded-2xl border border-legal-border bg-white">
          <div className="animate-pulse border-r border-slate-200 bg-slate-50" />
          <div className="animate-pulse bg-white" />
        </div>
      </div>
    );

  if (state.chaptersError)
    return (
      <div className="p-7">
        <div className="rounded-2xl border border-red-200 bg-red-50 p-7">
          <h2 className="text-xl font-bold text-red-800">
            Chapters could not be loaded
          </h2>
          <p className="mt-2 text-sm text-red-700">{state.chaptersError}</p>
          <button
            onClick={() => void state.retryChapters()}
            className="mt-5 flex items-center gap-2 rounded-lg bg-white px-4 py-2 text-sm font-semibold text-red-700 ring-1 ring-red-200"
          >
            <RefreshCw size={15} />
            Try again
          </button>
        </div>
      </div>
    );

  return (
    <div className="p-7">
      <div className="grid min-h-[650px] grid-cols-[330px_minmax(0,1fr)] overflow-hidden rounded-2xl border border-legal-border bg-white shadow-sm">
        <aside className="border-r border-legal-border bg-slate-50">
          <div className="border-b border-legal-border p-5">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs font-bold uppercase tracking-wider text-legal-muted">
                  BNS Chapters
                </div>
                <div className="mt-1 text-sm text-legal-muted">
                  {state.chapters.length} available
                </div>
              </div>
              <BookOpen className="text-legal-gold" size={23} />
            </div>
            <div className="mt-4 flex items-center rounded-xl border border-slate-300 bg-white px-3">
              <Search size={16} className="mr-2 text-slate-400" />
              <input
                value={filter}
                onChange={(event) => setFilter(event.target.value)}
                className="w-full py-2.5 text-sm outline-none"
                placeholder="Search chapters..."
              />
            </div>
          </div>
          <div className="max-h-[560px] overflow-y-auto p-3">
            {filtered.length ? (
              <div className="space-y-1">
                {filtered.map((chapter) => {
                  const active =
                    chapter.chapter_number === state.selectedChapter;
                  return (
                    <button
                      key={`${chapter.chapter_number}-${chapter.chapter_name}`}
                      onClick={() =>
                        state.setSelectedChapter(chapter.chapter_number)
                      }
                      className={`w-full rounded-xl p-3 text-left transition ${active ? "bg-legal-navy text-white shadow-sm" : "text-legal-text hover:bg-white"}`}
                    >
                      <div
                        className={`text-[10px] font-bold uppercase tracking-wider ${active ? "text-amber-300" : "text-legal-gold"}`}
                      >
                        Chapter {chapter.chapter_number}
                      </div>
                      <div className="mt-1 text-sm font-semibold leading-5">
                        {chapter.chapter_name}
                      </div>
                      <div
                        className={`mt-2 text-[11px] ${active ? "text-slate-300" : "text-legal-muted"}`}
                      >
                        {chapter.section_count} section
                        {chapter.section_count === 1 ? "" : "s"}
                      </div>
                    </button>
                  );
                })}
              </div>
            ) : (
              <div className="p-6 text-center text-sm text-legal-muted">
                No chapters match “{filter}”.
              </div>
            )}
          </div>
        </aside>

        <main className="min-w-0 p-7">
          {selected ? (
            <>
              <div className="flex items-start justify-between gap-5">
                <div>
                  <div className="text-xs font-bold uppercase tracking-wider text-legal-gold">
                    Chapter {selected.chapter_number}
                  </div>
                  <h2 className="mt-2 text-2xl font-bold text-legal-text">
                    {selected.chapter_name}
                  </h2>
                  <p className="mt-2 text-sm text-legal-muted">
                    {selected.section_count} section
                    {selected.section_count === 1 ? "" : "s"} in this chapter
                  </p>
                </div>
                <div className="grid size-12 place-items-center rounded-xl bg-amber-50 text-amber-700">
                  <BookOpen size={23} />
                </div>
              </div>
              <div className="my-6 h-px bg-slate-200" />
              <div className="mb-3 flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-legal-muted">
                <FileText size={15} />
                Sections
              </div>
              {state.sectionsLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3, 4].map((item) => (
                    <div
                      key={item}
                      className="h-16 animate-pulse rounded-xl bg-slate-100"
                    />
                  ))}
                </div>
              ) : state.sectionsError ? (
                <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                  <div>{state.sectionsError}</div>
                  <button
                    onClick={() => void state.retrySections()}
                    className="mt-3 flex items-center gap-2 font-semibold"
                  >
                    <RefreshCw size={14} />
                    Retry
                  </button>
                </div>
              ) : state.sections.length ? (
                <div className="divide-y divide-slate-100 rounded-xl border border-slate-200">
                  {state.sections.map((section) => (
                    <button
                      key={section.section_number}
                      onClick={() => onOpenSection(section.section_number)}
                      className="group flex w-full items-center justify-between gap-5 bg-white px-4 py-3 text-left first:rounded-t-xl last:rounded-b-xl hover:bg-slate-50"
                    >
                      <div className="min-w-0">
                        <div className="text-[10px] font-bold text-legal-gold">
                          SECTION {section.section_number}
                        </div>
                        <div className="mt-1 truncate text-sm font-semibold text-legal-text">
                          {section.section_name}
                        </div>
                      </div>
                      <ArrowRight
                        size={16}
                        className="shrink-0 text-slate-400 transition group-hover:translate-x-1 group-hover:text-legal-navy"
                      />
                    </button>
                  ))}
                </div>
              ) : (
                <div className="rounded-xl border border-dashed border-slate-300 p-8 text-center text-sm text-legal-muted">
                  No sections found for this chapter.
                </div>
              )}
            </>
          ) : (
            <div className="grid h-full place-items-center text-sm text-legal-muted">
              Select a chapter to view its sections.
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
