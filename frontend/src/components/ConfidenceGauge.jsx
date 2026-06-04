import React from 'react';
import { ShieldCheck, ShieldAlert, ShieldX } from 'lucide-react';

export default function ConfidenceGauge({ score = 0, reviewRequired = false }) {
  const pct = Math.round(score * 100);

  const { label, color, trackColor, Icon } =
    reviewRequired || pct < 70
      ? { label: 'REVIEW REQUIRED', color: 'text-red-400',    trackColor: 'bg-red-600',    Icon: ShieldX     }
    : pct < 85
      ? { label: 'MEDIUM',          color: 'text-amber-400',  trackColor: 'bg-amber-500',  Icon: ShieldAlert }
      : { label: 'HIGH',            color: 'text-emerald-400',trackColor: 'bg-emerald-500',Icon: ShieldCheck };

  return (
    <div className="flex items-center gap-2">
      <Icon size={13} className={color} />
      <div className="flex-1 bg-slate-700 rounded-full h-1.5 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${trackColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`text-[10px] font-bold ${color} shrink-0`}>
        G(r)={pct}% · {label}
      </span>
    </div>
  );
}
