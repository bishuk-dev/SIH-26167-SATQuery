import React, { useState } from 'react';

export const WorkspaceFooter = () => {
  const [showModal, setShowModal] = useState<string | null>(null);

  return (
    <>
      <footer className="h-10 border-t border-slate-800 bg-[#060b14] text-slate-300 flex items-center justify-between px-4 z-40 shrink-0 font-mono-data-sm text-xs">
      <div className="flex items-center gap-4 text-[11px] font-mono">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px_#34d399]"></span>
          <span className="font-bold text-emerald-400 tracking-wider">SYSTEM ONLINE</span>
          <div className="h-3 w-px bg-slate-800 mx-1"></div>
          <span className="text-slate-400">Satellite: <span className="text-slate-300">Sentinel-2</span></span>
          <div className="h-3 w-px bg-slate-800 mx-1"></div>
          <span className="text-slate-400">GSD: <span className="text-slate-300">10 m</span></span>
          <div className="h-3 w-px bg-slate-800 mx-1"></div>
          <span className="text-slate-400">AI: <span className="text-emerald-400 font-semibold">READY</span></span>
          <div className="h-3 w-px bg-slate-800 mx-1"></div>
          <span className="text-slate-400">Latency: <span className="text-slate-300">0.8s</span></span>
          <div className="h-3 w-px bg-slate-800 mx-1"></div>
          <span className="text-slate-400">CRS: <span className="text-slate-300">EPSG:32643</span></span>
        </div>
      </div>
      <div className="flex items-center gap-2 text-xs">
        <button onClick={() => setShowModal('telemetry')} className="px-2.5 py-1 text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent hover:border-slate-800 rounded font-mono-data-sm text-[11px] transition-colors flex items-center gap-1">
          <span className="material-symbols-outlined text-sm text-slate-400">terminal</span>
          <span>Telemetry Log</span>
        </button>
        <button onClick={() => setShowModal('radar')} className="px-2.5 py-1 text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent hover:border-slate-800 rounded font-mono-data-sm text-[11px] transition-colors flex items-center gap-1">
          <span className="material-symbols-outlined text-sm text-cyan-400">analytics</span>
          <span>Radar Histograms</span>
        </button>
      </div>
    </footer>

    {showModal && (
      <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setShowModal(null)}>
        <div className="bg-[#0a1222] border border-slate-700 p-6 rounded-xl shadow-2xl max-w-lg w-full text-slate-300" onClick={e => e.stopPropagation()}>
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-emerald-400 font-mono-data-sm font-bold text-lg">{showModal === 'telemetry' ? 'Telemetry Log' : 'Radar Histograms'}</h3>
            <button onClick={() => setShowModal(null)} className="text-slate-500 hover:text-white"><span className="material-symbols-outlined">close</span></button>
          </div>
          <p className="text-xs font-mono leading-relaxed bg-slate-950 p-4 rounded border border-slate-800 whitespace-pre-wrap">
            {showModal === 'telemetry' ? '[SYS] Retrieving archival datalink...\n[OK] Latency threshold: 842ms\n[OK] Projection verified: EPSG:32643\n[WARN] High variance in coastal bathymetry offset detected.' : 'Histogram Data:\n\nSigma-0 distribution peaks heavily at -12.4dB across targeted urban infrastructure zones. Dual-bounce scattering suggests heavy crane installations within the bounding box.'}
          </p>
        </div>
      </div>
    )}
    </>
  );
};
