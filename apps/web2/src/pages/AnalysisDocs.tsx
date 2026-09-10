import React from 'react';
import { PageShell } from '../components/PageShell';

export const AnalysisDocs = () => {
  return (
    <PageShell title="Analysis Documentation" subtitle="How SatQuery processes satellite data">
      <div className="max-w-4xl mx-auto space-y-10">

        {/* Section 1 */}
        <section className="rounded-2xl border border-white/[0.08] bg-white/[0.03] p-8">
          <h2 className="text-lg font-bold text-cyan-400 mb-4 flex items-center gap-2">
            <span className="w-6 h-6 rounded-md bg-cyan-500/10 border border-cyan-400/20 flex items-center justify-center text-xs font-mono text-cyan-400">1</span>
            Natural Language Spatial Queries
          </h2>
          <p className="text-slate-300 leading-relaxed mb-4">
            SatQuery allows you to interrogate multi-temporal satellite imagery using plain English. Our VQA (Visual Question Answering) engine translates your query into spatial reasoning tasks.
          </p>
          <div className="bg-black/50 p-4 rounded-xl border border-cyan-400/10 font-mono text-sm text-cyan-300">
            "Detect urban sprawl and lake boundary contraction in Bengaluru from 2022 to 2024"
          </div>
        </section>

        {/* Section 2 */}
        <section className="rounded-2xl border border-white/[0.08] bg-white/[0.03] p-8">
          <h2 className="text-lg font-bold text-cyan-400 mb-4 flex items-center gap-2">
            <span className="w-6 h-6 rounded-md bg-cyan-500/10 border border-cyan-400/20 flex items-center justify-center text-xs font-mono text-cyan-400">2</span>
            Supported Data Types
          </h2>
          <ul className="space-y-3">
            {[
              { label: 'Optical (RGB/NIR)', value: 'Sentinel-2, Landsat-8/9', color: 'emerald' },
              { label: 'SAR (Synthetic Aperture Radar)', value: 'Sentinel-1 C-band', color: 'sky' },
              { label: 'Elevation Models', value: 'Copernicus DEM, SRTM', color: 'amber' },
              { label: 'Formats', value: 'GeoTIFF, Cloud Optimized GeoTIFF (COG)', color: 'violet' },
            ].map(({ label, value, color }) => (
              <li key={label} className="flex items-center gap-3 text-sm">
                <span className={`w-2 h-2 rounded-full bg-${color}-400 flex-shrink-0`} />
                <span className="text-slate-400 w-56 flex-shrink-0">{label}:</span>
                <span className="text-slate-200 font-medium">{value}</span>
              </li>
            ))}
          </ul>
        </section>

        {/* Section 3 */}
        <section className="rounded-2xl border border-white/[0.08] bg-white/[0.03] p-8">
          <h2 className="text-lg font-bold text-cyan-400 mb-4 flex items-center gap-2">
            <span className="w-6 h-6 rounded-md bg-cyan-500/10 border border-cyan-400/20 flex items-center justify-center text-xs font-mono text-cyan-400">3</span>
            API Usage
          </h2>
          <p className="text-slate-300 mb-4 leading-relaxed">You can integrate SatQuery directly into your workflows using our REST API.</p>
          <pre className="bg-black/60 p-5 rounded-xl border border-white/[0.06] overflow-x-auto text-sm">
            <code className="text-sky-300 leading-relaxed">
              {`POST /api/vqa
Content-Type: application/json

{
  "observation_id": "obs_123456789",
  "question": "Count the number of anchored cargo ships."
}`}
            </code>
          </pre>
        </section>
      </div>
    </PageShell>
  );
};
