import { Construction } from "lucide-react";
export function Placeholder({ title }: { title: string }) {
  return (
    <div className="p-7">
      <div className="grid min-h-[420px] place-items-center rounded-2xl border border-dashed border-slate-300 bg-white">
        <div className="text-center">
          <div className="mx-auto grid size-14 place-items-center rounded-2xl bg-slate-100 text-slate-500">
            <Construction />
          </div>
          <h2 className="mt-4 text-xl font-bold text-legal-text">{title}</h2>
          <p className="mt-2 text-sm text-legal-muted">
            This module will be implemented in its Phase 4 step.
          </p>
        </div>
      </div>
    </div>
  );
}
