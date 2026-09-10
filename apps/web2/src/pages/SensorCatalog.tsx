import React from 'react';
import { PageShell } from '../components/PageShell';

const sensors = [
  { name: 'Sentinel-1', type: 'SAR (C-band)', resolution: '10m', revisit: '6-12 days', band: 'C', agency: 'ESA', color: 'emerald' },
  { name: 'Sentinel-2', type: 'Multispectral (MSI)', resolution: '10m – 60m', revisit: '5 days', band: '13 bands', agency: 'ESA', color: 'sky' },
  { name: 'Landsat-8', type: 'Multispectral (OLI)', resolution: '15m – 100m', revisit: '16 days', band: '11 bands', agency: 'USGS / NASA', color: 'amber' },
  { name: 'Landsat-9', type: 'Multispectral (OLI-2)', resolution: '15m – 100m', revisit: '16 days', band: '11 bands', agency: 'USGS / NASA', color: 'orange' },
  { name: 'MODIS', type: 'Multi-spectral', resolution: '250m – 1km', revisit: '1-2 days', band: '36 bands', agency: 'NASA', color: 'violet' },
  { name: 'Copernicus DEM', type: 'Elevation (DSM)', resolution: '30m', revisit: 'Static', band: 'Elevation', agency: 'ESA', color: 'rose' },
];

const colorMap: Record<string, string> = {
  emerald: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400',
  sky: 'bg-sky-500/10 border-sky-500/20 text-sky-400',
  amber: 'bg-amber-500/10 border-amber-500/20 text-amber-400',
  orange: 'bg-orange-500/10 border-orange-500/20 text-orange-400',
  violet: 'bg-violet-500/10 border-violet-500/20 text-violet-400',
  rose: 'bg-rose-500/10 border-rose-500/20 text-rose-400',
};

const dotMap: Record<string, string> = {
  emerald: 'bg-emerald-400', sky: 'bg-sky-400', amber: 'bg-amber-400',
  orange: 'bg-orange-400', violet: 'bg-violet-400', rose: 'bg-rose-400',
};

export const SensorCatalog = () => {
  return (
    <PageShell title="Sensor Catalog" subtitle="Supported satellite constellations and data sources">
      <div className="max-w-5xl mx-auto">
        <p className="text-slate-400 mb-8 leading-relaxed text-sm">
          SatQuery natively supports ingesting and reasoning over data from the following satellite constellations.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
          {sensors.map((sensor) => (
            <div
              key={sensor.name}
              className="rounded-2xl border border-white/[0.08] bg-white/[0.02] hover:bg-white/[0.05] p-6 transition-all group cursor-pointer"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-2.5">
                  <span className={`w-2.5 h-2.5 rounded-full ${dotMap[sensor.color]}`} />
                  <h3 className="text-[15px] font-bold text-white">{sensor.name}</h3>
                </div>
                <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${colorMap[sensor.color]}`}>
                  {sensor.agency}
                </span>
              </div>
              <p className="text-xs text-slate-400 mb-4">{sensor.type}</p>
              <div className="space-y-2 text-xs font-mono">
                {[
                  { label: 'Resolution', value: sensor.resolution },
                  { label: 'Revisit Time', value: sensor.revisit },
                  { label: 'Bands', value: sensor.band },
                ].map(({ label, value }) => (
                  <div key={label} className="flex justify-between text-slate-500">
                    <span>{label}</span>
                    <span className="text-slate-200">{value}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </PageShell>
  );
};
