import React from 'react';
import { PageShell } from '../components/PageShell';

const metrics = [
  { label: 'VQA Accuracy (Optical)', value: '89.4%', color: 'emerald', desc: 'Tested on EuroSAT + UC Merced datasets' },
  { label: 'VQA Accuracy (SAR)',     value: '82.1%', color: 'sky',     desc: 'Tested on Sentinel-1 GRD scenes' },
  { label: 'Average Latency',        value: '1.2s',  color: 'amber',   desc: 'p50 response time across 10k queries' },
  { label: 'Change Detection mIoU',  value: '78.6%', color: 'violet',  desc: 'Multi-temporal Sentinel-2 stack' },
  { label: 'Flood Mapping F1',       value: '91.3%', color: 'blue',    desc: 'Bangladesh & Kerala flood events' },
  { label: 'Throughput',             value: '340/hr', color: 'rose',   desc: 'Concurrent VQA requests' },
];

const colorMap: Record<string, string> = {
  emerald: 'text-emerald-400', sky: 'text-sky-400', amber: 'text-amber-400',
  violet: 'text-violet-400', blue: 'text-blue-400', rose: 'text-rose-400',
};
const borderMap: Record<string, string> = {
  emerald: 'border-emerald-500/20', sky: 'border-sky-500/20', amber: 'border-amber-500/20',
  violet: 'border-violet-500/20', blue: 'border-blue-500/20', rose: 'border-rose-500/20',
};

export const Benchmarks = () => {
  return (
    <PageShell title="Benchmarks" subtitle="Performance metrics and accuracy for SatQuery VQA models">
      <div className="max-w-5xl mx-auto space-y-10">

        {/* Metrics grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {metrics.map(({ label, value, color, desc }) => (
            <div key={label} className={`rounded-2xl border ${borderMap[color]} bg-white/[0.02] p-6 text-center hover:bg-white/[0.04] transition-all`}>
              <div className={`text-4xl font-extrabold tracking-tight mb-1 ${colorMap[color]}`}>{value}</div>
              <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest mb-2">{label}</div>
              <div className="text-[10px] text-slate-600 leading-relaxed">{desc}</div>
            </div>
          ))}
        </div>

        {/* Architecture */}
        <section className="rounded-2xl border border-white/[0.08] bg-white/[0.02] p-8">
          <h2 className="text-base font-bold text-white mb-5 flex items-center gap-2">
            <svg className="w-4 h-4 text-cyan-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <rect height="16" rx="2" width="16" x="4" y="4" />
              <rect height="6" width="6" x="9" y="9" />
            </svg>
            Model Architecture
          </h2>
          <div className="space-y-4 text-sm text-slate-300 leading-relaxed">
            <p>
              SatQuery utilizes a fine-tuned multimodal architecture capable of directly fusing spatial tokens from Sentinel-1/2 rasters with linguistic embeddings.
            </p>
            <p>
              Unlike standard LLaVA models, our system natively understands coordinate reference systems (CRS) and radiometric calibrations, enabling true change detection over multi-temporal stacks rather than just static image captioning.
            </p>
          </div>

          {/* Architecture diagram pills */}
          <div className="mt-6 flex flex-wrap items-center gap-2 text-xs">
            {['RS-LLaVA Encoder', '→', 'Spatial Token Fusion', '→', 'Temporal Attention', '→', 'Change Detection Head'].map((item, i) => (
              item === '→'
                ? <span key={i} className="text-slate-600 font-mono">→</span>
                : <span key={i} className="px-3 py-1.5 rounded-lg bg-cyan-500/[0.07] border border-cyan-400/15 text-cyan-300 font-medium">{item}</span>
            ))}
          </div>
        </section>

        {/* Dataset table */}
        <section className="rounded-2xl border border-white/[0.08] bg-white/[0.02] p-8">
          <h2 className="text-base font-bold text-white mb-5">Evaluation Datasets</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-slate-300">
              <thead className="text-[11px] uppercase text-slate-500 border-b border-white/[0.07]">
                <tr>
                  <th className="text-left pb-3">Dataset</th>
                  <th className="text-left pb-3">Task</th>
                  <th className="text-left pb-3">Samples</th>
                  <th className="text-right pb-3">Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {[
                  { ds: 'EuroSAT', task: 'Land Use Classification', n: '27,000', score: '96.2%' },
                  { ds: 'UC Merced', task: 'Scene Recognition (VQA)', n: '21,000', score: '89.4%' },
                  { ds: 'FloodNet', task: 'Post-flood Damage Assessment', n: '2,343', score: '91.3%' },
                  { ds: 'DIOR-RSVG', task: 'Visual Grounding', n: '17,402', score: '74.8%' },
                ].map(({ ds, task, n, score }) => (
                  <tr key={ds} className="hover:bg-white/[0.02] transition">
                    <td className="py-3 font-medium text-white">{ds}</td>
                    <td className="py-3 text-slate-400">{task}</td>
                    <td className="py-3 text-slate-500 font-mono text-xs">{n}</td>
                    <td className="py-3 text-right font-bold text-emerald-400">{score}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </PageShell>
  );
};
