import React from 'react';
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
} from 'recharts';
import { MetricsSummary } from '../api/types';

interface ChartsSectionProps {
  metrics: MetricsSummary | null;
}

const CACHE_COLORS = ['#10b981', '#f59e0b']; // emerald, amber
const MODEL_COLORS = ['#0284c7', '#8b5cf6', '#06b6d4']; // sky, purple, cyan

export const ChartsSection: React.FC<ChartsSectionProps> = ({ metrics }) => {
  if (!metrics || metrics.total_requests === 0) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 flex flex-col items-center justify-center min-h-[220px] text-center">
          <p className="text-xs font-mono text-slate-400">No cache telemetry data yet</p>
          <span className="text-[11px] text-slate-400 mt-1">Run queries above to populate cache performance analytics</span>
        </div>
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 flex flex-col items-center justify-center min-h-[220px] text-center">
          <p className="text-xs font-mono text-slate-400">No model routing data yet</p>
          <span className="text-[11px] text-slate-400 mt-1">Routing decisions will appear here as requests are dispatched</span>
        </div>
      </div>
    );
  }

  // Cache breakdown data
  const cacheData = [
    { name: 'Cache Hits', value: metrics.cache_hits },
    { name: 'Cache Misses', value: metrics.cache_misses },
  ].filter((d) => d.value > 0);

  // Model breakdown data
  const modelData = Object.entries(metrics.routing_breakdown || {}).map(([model, count]) => ({
    name: model,
    count,
  }));

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
      {/* Cache Hit vs Miss Chart */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Cache Outcome Breakdown
            </h3>
            <p className="text-[11px] text-slate-400 font-mono mt-0.5">
              Hit Rate: {(metrics.cache_hit_rate * 100).toFixed(1)}% ({metrics.cache_hits} hits / {metrics.total_requests} reqs)
            </p>
          </div>
        </div>

        <div className="h-48 w-full flex items-center justify-center">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={cacheData}
                cx="50%"
                cy="50%"
                innerRadius={45}
                outerRadius={70}
                paddingAngle={4}
                dataKey="value"
              >
                {cacheData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.name === 'Cache Hits' ? CACHE_COLORS[0] : CACHE_COLORS[1]}
                  />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: '#090d16',
                  borderColor: '#1e293b',
                  borderRadius: '0.375rem',
                  fontSize: '12px',
                  fontFamily: 'JetBrains Mono',
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="flex items-center justify-center space-x-6 text-xs font-mono mt-2">
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
            <span className="text-slate-300">Hits ({metrics.cache_hits})</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
            <span className="text-slate-300">Misses ({metrics.cache_misses})</span>
          </div>
        </div>
      </div>

      {/* Model Routing Distribution Chart */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Model Routing Distribution
            </h3>
            <p className="text-[11px] text-slate-400 font-mono mt-0.5">
              Traffic allocated across model tiers by complexity
            </p>
          </div>
        </div>

        <div className="h-48 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={modelData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
              <XAxis
                dataKey="name"
                stroke="#64748b"
                fontSize={10}
                fontFamily="JetBrains Mono"
                tickLine={false}
              />
              <YAxis
                stroke="#64748b"
                fontSize={10}
                fontFamily="JetBrains Mono"
                allowDecimals={false}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#090d16',
                  borderColor: '#1e293b',
                  borderRadius: '0.375rem',
                  fontSize: '12px',
                  fontFamily: 'JetBrains Mono',
                }}
              />
              <Bar dataKey="count" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="text-center text-[11px] text-slate-400 font-mono mt-2">
          Cost-aware routing prioritizes lightweight models unless reasoning heuristics demand frontier capacity.
        </div>
      </div>
    </div>
  );
};
