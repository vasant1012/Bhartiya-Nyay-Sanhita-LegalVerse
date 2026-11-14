import {
  Bookmark,
  BookOpen,
  Compass,
  Info,
  Scale,
  Search,
  Sparkles,
} from "lucide-react";

export type Page =
  | "explore"
  | "search"
  | "chapters"
  | "insights"
  | "bookmarks"
  | "about";

const nav = [
  { id: "explore", label: "Explore", icon: Compass },
  { id: "search", label: "Smart Search", icon: Search },
  { id: "chapters", label: "Chapters", icon: BookOpen },
  { id: "insights", label: "Insights", icon: Sparkles },
  { id: "bookmarks", label: "Bookmarks", icon: Bookmark },
] as const;

export function Sidebar({
  page,
  onChange,
}: {
  page: Page;
  onChange: (page: Page) => void;
}) {
  return (
    <aside className="flex h-screen w-64 shrink-0 flex-col bg-legal-deep px-4 py-5 text-white shadow-xl">
      <div className="flex items-center gap-3 px-2 py-2">
        <div className="grid size-10 place-items-center rounded-xl bg-legal-gold shadow-lg shadow-black/10">
          <Scale size={23} />
        </div>
        <div>
          <div className="text-xl font-bold tracking-tight">LegalVerse</div>
          <div className="text-xs text-slate-300">BNS Intelligence</div>
        </div>
      </div>
      <div className="mt-8 flex-1 space-y-1">
        {nav.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => onChange(id)}
            className={`flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm transition ${page === id ? "bg-white/15 font-semibold text-white" : "text-slate-300 hover:bg-white/8 hover:text-white"}`}
          >
            <Icon size={19} />
            <span>{label}</span>
          </button>
        ))}
      </div>
      <button
        onClick={() => onChange("about")}
        className={`flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm ${page === "about" ? "bg-white/15 text-white" : "text-slate-300 hover:bg-white/8"}`}
      >
        <Info size={19} />
        About
      </button>
      <div className="mt-4 border-t border-white/10 px-3 pt-4 text-[11px] leading-4 text-slate-400">
        Research and educational demo
        <br />
        Not legal advice
      </div>
    </aside>
  );
}
