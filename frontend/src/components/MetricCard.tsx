import React from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subValue?: string;
  badge?: string;
  badgeType?: 'default' | 'success' | 'warning' | 'info';
  icon: React.ReactNode;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subValue,
  badge,
  badgeType = 'default',
  icon,
}) => {
  const badgeColors = {
    default: 'bg-slate-800 text-slate-400 border-slate-700',
    success: 'bg-emerald-950/60 text-emerald-400 border-emerald-800/60',
    warning: 'bg-amber-950/60 text-amber-400 border-amber-800/60',
    info: 'bg-sky-950/60 text-sky-400 border-sky-800/60',
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 hover:border-slate-700/80 transition-all flex flex-col justify-between">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
          {title}
        </span>
        <div className="p-2 bg-slate-800/60 border border-slate-700/40 rounded-lg text-slate-400">
          {icon}
        </div>
      </div>

      <div className="mt-4">
        <div className="flex items-baseline space-x-2">
          <span className="text-2xl font-bold tracking-tight text-slate-100 font-mono">
            {value}
          </span>
          {badge && (
            <span
              className={`text-[10px] font-mono px-2 py-0.5 rounded border ${badgeColors[badgeType]}`}
            >
              {badge}
            </span>
          )}
        </div>
        {subValue && (
          <p className="mt-1 text-xs text-slate-400 font-mono">
            {subValue}
          </p>
        )}
      </div>
    </div>
  );
};
