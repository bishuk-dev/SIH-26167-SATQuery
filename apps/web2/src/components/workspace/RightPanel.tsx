import React, { useRef, useState } from 'react';

interface RightPanelProps {
  opticalOpacity: number;
  setOpticalOpacity: (val: number) => void;
  diffActive: boolean;
  toggleDiff: () => void;
  setObservationId: (id: string) => void;
}

export const RightPanel: React.FC<RightPanelProps> = ({ opticalOpacity, setOpticalOpacity, diffActive, toggleDiff, setObservationId }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://localhost:8000/api/observations', {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        const data = await res.json();
        setObservationId(data.observation_id);
        alert(`Ingested successfully. Observation ID: ${data.observation_id}`);
      } else {
        alert('Upload failed.');
      }
    } catch (err: any) {
      alert(`Upload failed: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };
  return (
    <aside className="w-80 border-l border-slate-800 bg-[#080e1a] text-slate-200 flex flex-col z-20 shrink-0 select-none">
      <div className="p-3 border-b border-slate-800 bg-[#0a1222]">
        <div className="flex items-center justify-between mb-2.5">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-base text-cyan-400">layers</span>
            <span className="font-headline-sm text-xs font-bold uppercase tracking-wider text-slate-100">Raster Layers</span>
            <span className="text-[9px] font-mono-data-sm px-1.5 py-[2px] rounded bg-slate-900 text-slate-400 border border-slate-800">STAC_V1</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] font-mono-data-sm text-slate-400">2 INGESTS</span>
            <button className="p-1 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded transition-colors flex items-center justify-center" title="Collapse Panel">
              <span className="material-symbols-outlined text-base">dock_to_right</span>
            </button>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-1 bg-[#060b14] p-1 rounded border border-slate-800 text-xs font-mono-data-sm">
          <button className="py-1 text-center bg-cyan-600/20 text-cyan-300 border border-cyan-500/40 rounded font-semibold text-[11px] transition-all">ACTIVE (2)</button>
          <button className="py-1 text-center text-slate-400 hover:text-slate-200 rounded text-[11px] transition-all">SPECTRAL</button>
          <button className="py-1 text-center text-slate-400 hover:text-slate-200 rounded text-[11px] transition-all">GEO-REF</button>
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto p-3 space-y-3 custom-scroll">
        <div className="rounded border border-emerald-500/30 bg-[#0a1426] p-3 space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_6px_#34d399]"></span>
              <span className="font-mono-data-sm text-xs font-semibold text-slate-100">OPTICAL BASELINE [T0]</span>
            </div>
            <span className="text-[10px] font-mono-data-sm px-1.5 py-[2px] rounded bg-emerald-950 text-emerald-300 border border-emerald-500/40 font-bold">18-APR-2023</span>
          </div>
          <div className="text-xs font-mono-data-sm text-slate-400 space-y-1.5 pt-1 border-t border-slate-800/80">
            <div className="flex justify-between"><span>SENSOR / PLATFORM</span><span className="text-slate-200 font-semibold">Sentinel-2B MSI</span></div>
            <div className="flex justify-between"><span>BANDS UTILIZED</span><span className="text-emerald-400 font-semibold">B4(R), B3(G), B2(B), B8(NIR)</span></div>
            <div className="flex justify-between"><span>GSD RESOLUTION</span><span className="text-slate-200">10.0m / pixel</span></div>
            <div className="flex justify-between"><span>ATMOSPHERE / CLOUD</span><span className="text-emerald-400 font-semibold">0.4% (Clear, BOA L2A)</span></div>
          </div>
        </div>
        
        <div className="rounded border border-cyan-500/30 bg-[#0a1426] p-3 space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_6px_#38bdf8]"></span>
              <span className="font-mono-data-sm text-xs font-semibold text-slate-100">SAR RADAR [T1]</span>
            </div>
            <span className="text-[10px] font-mono-data-sm px-1.5 py-[2px] rounded bg-cyan-950 text-cyan-300 border border-cyan-500/40 font-bold">14-MAR-2024</span>
          </div>
          <div className="text-xs font-mono-data-sm text-slate-400 space-y-1.5 pt-1 border-t border-slate-800/80">
            <div className="flex justify-between"><span>RADAR INSTRUMENT</span><span className="text-slate-200 font-semibold">Sentinel-1 C-Band SAR</span></div>
            <div className="flex justify-between"><span>POLARIZATION</span><span className="text-cyan-400 font-semibold">VV + VH Dual-Pol</span></div>
            <div className="flex justify-between"><span>PASS / INCIDENCE</span><span className="text-slate-200">Ascending (38.2° θ)</span></div>
            <div className="flex justify-between"><span>BACKSCATTER SHIFT</span><span className="text-cyan-300 font-bold">+4.2 dB Sigma-0</span></div>
          </div>
        </div>

        <div className="rounded border border-slate-800 bg-[#08101e] p-3 space-y-3">
          <div className="flex items-center justify-between text-xs font-mono-data-sm">
            <span className="font-medium text-slate-300 uppercase">[SPLIT_BLENDING]</span>
            <span className="text-cyan-400 font-bold">{opticalOpacity}% OPTICAL</span>
          </div>
          <input 
            className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-400" 
            type="range" min="0" max="100" 
            value={opticalOpacity}
            onChange={(e) => setOpticalOpacity(parseInt(e.target.value))}
          />
          <div className="flex justify-between text-[9px] font-mono-data-sm text-slate-500">
            <span>0% (PURE SAR)</span>
            <span>50% (SWIPE)</span>
            <span>100% (S2 MSI)</span>
          </div>
          <div className="flex items-center justify-between pt-2 border-t border-slate-800 text-xs">
            <div className="flex items-center gap-1.5 text-slate-200 font-mono-data-sm text-[11px]">
              <span className="material-symbols-outlined text-sm text-amber-400">dynamic_feed</span>
              <span>HIGHLIGHT CHANGE ANOMALIES</span>
            </div>
            <button 
              className={`w-8 h-4 rounded-full relative transition-colors ${diffActive ? 'bg-amber-600/80' : 'bg-slate-700'}`} 
              onClick={toggleDiff}
            >
              <span className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all shadow-sm ${diffActive ? 'left-4' : 'left-[2px]'}`}></span>
            </button>
          </div>
        </div>

        <div className="pt-1 space-y-2">
          <span className="text-[10px] font-mono-data-sm font-semibold uppercase tracking-wider text-slate-400">[FORENSIC_GIS_OVERLAYS]</span>
          <div className="space-y-1 text-xs font-mono-data-sm">
            <label className="flex items-center justify-between text-slate-200 cursor-pointer py-1.5 px-2 hover:bg-slate-800/60 rounded border border-transparent hover:border-slate-700/60">
              <div className="flex items-center gap-2">
                <input defaultChecked className="rounded border-slate-700 bg-slate-900 text-cyan-500 focus:ring-0 w-3.5 h-3.5" type="checkbox" />
                <span>Terminals &amp; Port Cadastre</span>
              </div>
              <span className="text-cyan-400 text-[11px] font-bold">14 ZONES</span>
            </label>
            <label className="flex items-center justify-between text-slate-200 cursor-pointer py-1.5 px-2 hover:bg-slate-800/60 rounded border border-transparent hover:border-slate-700/60">
              <div className="flex items-center gap-2">
                <input defaultChecked className="rounded border-slate-700 bg-slate-900 text-violet-500 focus:ring-0 w-3.5 h-3.5" type="checkbox" />
                <span>Navigational Corridors</span>
              </div>
              <span className="text-violet-400 text-[11px] font-bold">3 LANES</span>
            </label>
            <label className="flex items-center justify-between text-slate-200 cursor-pointer py-1.5 px-2 hover:bg-slate-800/60 rounded border border-transparent hover:border-slate-700/60">
              <div className="flex items-center gap-2">
                <input defaultChecked className="rounded border-slate-700 bg-slate-900 text-amber-500 focus:ring-0 w-3.5 h-3.5" type="checkbox" />
                <span>Identified Unpermitted Fill</span>
              </div>
              <span className="text-amber-400 text-[11px] font-bold">+34.8k m²</span>
            </label>
          </div>
        </div>
      </div>
      
      <div className="p-3 border-t border-slate-800 bg-[#0a1222]">
        <input 
          type="file" 
          ref={fileInputRef} 
          style={{ display: 'none' }} 
          accept=".tif,.tiff" 
          onChange={handleFileChange}
        />
        <button 
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
          className="w-full py-2 bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-700 hover:border-cyan-500/60 font-mono-data-sm text-xs font-semibold rounded flex items-center justify-center gap-1.5 transition-colors disabled:opacity-50"
        >
          <span className={`material-symbols-outlined text-sm text-cyan-400 ${isUploading ? 'animate-spin' : ''}`}>
            {isUploading ? 'refresh' : 'add_circle'}
          </span>
          <span>{isUploading ? 'INGESTING...' : 'INGEST ARCHIVAL COPERNICUS'}</span>
        </button>
      </div>
    </aside>
  );
};
