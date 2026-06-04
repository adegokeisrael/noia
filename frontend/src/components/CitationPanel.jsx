import React, { useState } from 'react';
import { ChevronDown, ChevronUp, BookOpen } from 'lucide-react';

const TYPE_COLORS = {
  runbook:     'bg-blue-900 text-blue-300 border-blue-700',
  incident:    'bg-red-900 text-red-300 border-red-700',
  sla:         'bg-amber-900 text-amber-300 border-amber-700',
  maintenance: 'bg-emerald-900 text-emerald-300 border-emerald-700',
};

export default function CitationPanel({ sources = [] }) {
  const [open, setOpen] = useState(false);
  if (!sources.length) return null;

  return (
    <div className="mt-1">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200
                   transition-colors duration-150"
      >
        <BookOpen size={11} />
        {sources.length} source{sources.length > 1 ? 's' : ''} retrieved
        {open ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
      </button>

      {open && (
        <div className="mt-2 space-y-1.5">
          {sources.map((src) => (
            <div
              key={src.source_num}
              className="flex items-start gap-2 bg-slate-900 border border-slate-700
                         rounded-lg px-3 py-2"
            >
              <span className="text-slate-500 text-xs font-mono mt-0.5 shrink-0">
                [{src.source_num}]
              </span>
              <div className="min-w-0">
                <p className="text-xs text-slate-200 font-medium truncate">{src.title}</p>
                <div className="flex items-center gap-2 mt-0.5 flex-wrap">
                  <span className={`text-[10px] font-semibold border rounded px-1.5 py-0.5
                                   ${TYPE_COLORS[src.source_type] || 'bg-slate-700 text-slate-300 border-slate-600'}`}>
                    {src.source_type?.toUpperCase()}
                  </span>
                  <span className="text-[10px] text-slate-500">{src.timestamp}</span>
                  <span className="text-[10px] text-slate-500">
                    score: {(src.score * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
