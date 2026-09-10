import React from 'react';

export const WorkspaceHeader = () => {
  const handleShare = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      alert('Link copied to clipboard!');
    } catch (err) {
      alert('Failed to copy link.');
    }
  };

  const handleExport = () => {
    const mockData = JSON.stringify({ 
      type: "FeatureCollection", 
      features: [
        { type: "Feature", properties: { target: "Bulkhead Reclamation", area: 34800 }, geometry: { type: "Point", coordinates: [2.154, 41.358] } }
      ] 
    }, null, 2);
    const blob = new Blob([mockData], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'satquery_export.geojson';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <header className="flex justify-between items-center w-full px-4 h-12 border-b border-slate-800 bg-[#070d18] text-slate-200 z-50 shrink-0 font-body-sm">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-md bg-emerald-950/60 border border-emerald-500/40 flex items-center justify-center text-emerald-400 shadow-[0_0_10px_rgba(16,185,129,0.2)]">
            <span className="material-symbols-outlined text-lg">satellite_alt</span>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-headline-sm text-sm font-bold tracking-wider text-slate-100 uppercase">SatQuery</span>
              <span className="px-1.5 py-0.5 rounded bg-emerald-950/80 border border-emerald-500/40 text-[10px] font-mono-data-sm text-emerald-400 tracking-wider font-semibold">GEO·INT PRO</span>
            </div>
            <p className="text-[10px] font-mono-data-sm text-slate-400 leading-none">CO-REGISTERED EO &amp; SAR INTELLIGENCE</p>
          </div>
        </div>
        <div className="h-5 w-px bg-slate-800 hidden sm:block"></div>
        <div className="hidden md:flex items-center gap-2 text-xs font-mono-data-sm">
          <span className="text-slate-400">PROJECT:</span>
          <span className="font-medium text-slate-200">PORT_EXPANSION // BCN</span>
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 shadow-[0_0_6px_#38bdf8]"></span>
          <span className="text-cyan-300 text-[11px]">S2-MSI &amp; S1-SAR C-BAND</span>
          <span className="text-slate-400 ml-1 text-[10px] px-1.5 py-[2px] bg-slate-900 border border-slate-800 rounded">ORBIT 142</span>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <div className="hidden lg:flex items-center gap-2 px-2.5 py-1 rounded bg-slate-900/90 border border-emerald-500/30 text-emerald-300 text-xs font-mono-data-sm">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
          <span className="font-semibold text-emerald-400">STATUS:</span>
          <span className="text-slate-300">2 RASTERS SYNCED (ΔT 330d)</span>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={handleShare} className="px-3 py-1 rounded border border-slate-700 bg-slate-900/60 hover:border-slate-500 hover:bg-slate-800 text-slate-300 text-xs font-mono-data-sm flex items-center gap-1.5 transition-colors">
            <span className="material-symbols-outlined text-sm text-slate-400">share</span>
            <span>Share View</span>
          </button>
          <button onClick={handleExport} className="px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white font-mono-data-sm text-xs font-semibold rounded active:scale-95 flex items-center gap-1.5 shadow-[0_0_12px_rgba(16,185,129,0.35)] transition-all">
            <span className="material-symbols-outlined text-sm font-semibold">download</span>
            <span>Export GEO-JSON / PDF</span>
          </button>
        </div>
      </div>
    </header>
  );
};
