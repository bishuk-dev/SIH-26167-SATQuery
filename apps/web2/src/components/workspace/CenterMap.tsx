import React, { useState, useRef, useEffect } from 'react';

interface CenterMapProps {
  opticalOpacity: number;
  diffActive: boolean;
  activeTarget: string | null;
}

export const CenterMap: React.FC<CenterMapProps> = ({ opticalOpacity, diffActive, activeTarget }) => {
  const [sliderPos, setSliderPos] = useState(50);
  const [isDraggingSlider, setIsDraggingSlider] = useState(false);
  const [scale, setScale] = useState(1);
  const [panX, setPanX] = useState(0);
  const [panY, setPanY] = useState(0);
  const [isPanning, setIsPanning] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });
  const [isFullscreen, setIsFullscreen] = useState(false);
  const viewportRef = useRef<HTMLDivElement>(null);

  const updateSplit = (clientX: number) => {
    if (!viewportRef.current) return;
    const rect = viewportRef.current.getBoundingClientRect();
    let posX = clientX - rect.left;
    posX = Math.max(0, Math.min(posX, rect.width));
    const percentage = (posX / rect.width) * 100;
    setSliderPos(percentage);
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDraggingSlider) {
        updateSplit(e.clientX);
      } else if (isPanning && scale > 1) {
        setPanX((prev) => prev + (e.clientX - lastMousePos.x));
        setPanY((prev) => prev + (e.clientY - lastMousePos.y));
        setLastMousePos({ x: e.clientX, y: e.clientY });
      }
    };

    const handleMouseUp = () => {
      if (isDraggingSlider) setIsDraggingSlider(false);
      if (isPanning) setIsPanning(false);
    };

    if (isDraggingSlider || isPanning) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDraggingSlider, isPanning, lastMousePos, scale]);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch((err) => {
        console.error(`Error attempting to enable fullscreen mode: ${err.message}`);
      });
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
  };

  const getTargetStyle = (targetId: string) => {
    let classes = "absolute z-20 cursor-pointer group transition-all duration-700 ";
    if (!diffActive) {
      classes += "opacity-30 ";
    }
    if (activeTarget === targetId) {
      classes += "scale-125 brightness-150 ";
    }
    return classes;
  };

  return (
    <section className="flex-1 relative bg-surface-container-lowest overflow-hidden flex items-center justify-center">
      <div 
        ref={viewportRef}
        className={`relative w-full h-full select-none overflow-hidden ${isDraggingSlider ? 'cursor-ew-resize' : (scale > 1 ? (isPanning ? 'cursor-grabbing' : 'cursor-grab') : 'cursor-crosshair')}`}
        onMouseDown={(e) => {
          if (!isDraggingSlider && scale > 1) {
            setIsPanning(true);
            setLastMousePos({ x: e.clientX, y: e.clientY });
          }
        }}
        onWheel={(e) => {
          const delta = e.deltaY > 0 ? -0.2 : 0.2;
          setScale(s => {
            const newScale = Math.min(Math.max(s + delta, 1), 5);
            if (newScale === 1) {
              setPanX(0);
              setPanY(0);
            }
            return newScale;
          });
        }}
      >
        {/* Background SAR Image */}
        <div className="absolute inset-0 w-full h-full">
          <div 
            className="absolute inset-0 w-full h-full origin-center transition-transform duration-200 ease-out"
            style={{ transform: `translate(${panX}px, ${panY}px) scale(${scale})` }}
          >
          <img 
            alt="Sentinel-1 SAR C-band Radar Co-registered Imagery of Coastal Port Complex" 
            className="w-full h-full object-cover select-none pointer-events-none filter brightness-95 contrast-125" 
            src="https://lh3.googleusercontent.com/aida-public/AB6AXuAWmmwfHKFapFDRx7ZTeNTDzn5HQhHJevVJWc-pSnSuC65cCjmHV8cJ1yTChLFYs0TEyUaWRl1OmIt2TWABzfQpELxt6kJfqtCmAmn4icG5jqRTRgO77YWxUwBgsvkGz6voq5a8ujZ9p8N39Qo53N0RRnEvJyfw2y1q-KKWH--6dPPVhlP5HtcJF_4fQuC22WiAY1IiCkgrPh0Zri0vzE73xUpj6u8LuOrZlnJ-6nk-U3yMFXtZwN2H"
          />
          <div className="absolute top-4 right-4 bg-[#070e1b]/90 backdrop-blur border border-cyan-500/40 px-3 py-1.5 rounded text-cyan-300 font-mono-data-sm text-xs z-10 flex items-center gap-2 shadow-xl">
            <span className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_6px_#38bdf8]"></span>
            <span className="font-bold">T1 // Sentinel-1 SAR (2024)</span>
            <span className="text-[10px] text-slate-400 px-1 rounded bg-slate-900 border border-slate-800">C-BAND</span>
          </div>
          </div>
        </div>

        {/* Clipped Optical Overlay */}
        <div 
          className="absolute inset-0 w-full h-full overflow-hidden pointer-events-none"
          style={{ clipPath: `inset(0px ${100 - sliderPos}% 0px 0px)` }}
        >
          <div 
            className="absolute inset-0 w-full h-full origin-center transition-transform duration-200 ease-out"
            style={{ transform: `translate(${panX}px, ${panY}px) scale(${scale})` }}
          >
          <img 
            alt="Sentinel-2 True Color Optical Satellite Ingest of Coastal Port Infrastructure" 
            className="absolute inset-0 w-full h-full object-cover select-none pointer-events-none transition-opacity duration-200" 
            style={{ opacity: opticalOpacity / 100 }}
            src="https://lh3.googleusercontent.com/aida-public/AB6AXuAz1XubugP803YuP4ImSzKLA7CDb9oU76IoAAmWASyMdZL3UMEj43mwOz3Vm8PBs_hQwQUbuYhSGv4hLcX-ln9fWOJ6QnPMFn6XRISRmDqfctH5Wy2LuYmAg_B7m9Pi_8LtqJU44PLvoV3xgS2GhKnxl7KDtYNNvIEP7R_xnEKkVyYZaurNRiPuquUlBIpXr5HkarBvNnH5hcbkLscAv_75hIGS_xneFINMH1nsB4u7BknJK5h9qzXH"
          />
          <div className="absolute top-4 left-4 bg-[#070e1b]/90 backdrop-blur border border-emerald-500/40 px-3 py-1.5 rounded text-emerald-300 font-mono-data-sm text-xs z-10 flex items-center gap-2 shadow-xl pointer-events-auto">
            <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_6px_#34d399]"></span>
            <span className="font-bold">T0 // Sentinel-2 Optical (2023)</span>
            <span className="text-[10px] text-slate-400 px-1 rounded bg-slate-900 border border-slate-800">RGB-NIR</span>
          </div>
          </div>
        </div>

        {/* Target Markers Overlay Layer */}
        <div 
          className="absolute inset-0 w-full h-full origin-center transition-transform duration-200 ease-out pointer-events-none"
          style={{ transform: `translate(${panX}px, ${panY}px) scale(${scale})` }}
        >

        {/* Target Alpha */}
        <div className={`${getTargetStyle('targetAlpha')} top-[48%] left-[42%] pointer-events-auto`}>
          <div className="relative">
            <span className="w-5 h-5 rounded-full bg-amber-500 border-2 border-slate-900 flex items-center justify-center text-[10px] font-bold text-slate-950 shadow-[0_0_10px_#fbbf24] font-mono-data-sm">01</span>
            <div className="absolute left-6 -top-3 w-52 p-2.5 rounded bg-[#070d18]/95 backdrop-blur border border-amber-500/50 shadow-2xl pointer-events-auto transition-transform group-hover:scale-105 opacity-0 invisible group-hover:opacity-100 group-hover:visible">
              <div className="flex items-center justify-between text-amber-400 font-mono-data-sm text-xs font-bold mb-1">
                <span>Bulkhead Reclamation</span>
                <span className="text-[10px] bg-amber-950/90 text-amber-300 border border-amber-500/40 px-1.5 py-[2px] rounded font-mono-data-sm">+34,800m²</span>
              </div>
              <p className="text-[11px] font-body-sm text-slate-300 leading-snug">Unpermitted shoreline infill identified between T0 &amp; T1.</p>
              <div className="mt-1 text-[10px] font-mono-data-sm text-amber-300/90">ANOMALY: COASTAL BOUNDARY SHIFT</div>
            </div>
          </div>
        </div>

        {/* Target Beta */}
        <div className={`${getTargetStyle('targetBeta')} top-[28%] left-[58%] pointer-events-auto`}>
          <div className="relative">
            <span className="w-5 h-5 rounded-full bg-cyan-400 border-2 border-slate-900 flex items-center justify-center text-[10px] font-bold text-slate-950 shadow-[0_0_10px_#38bdf8] font-mono-data-sm">02</span>
            <div className="absolute left-6 -top-3 w-52 p-2.5 rounded bg-[#070d18]/95 backdrop-blur border border-cyan-500/50 shadow-2xl pointer-events-auto transition-transform group-hover:scale-105 opacity-0 invisible group-hover:opacity-100 group-hover:visible">
              <div className="flex items-center justify-between text-cyan-300 font-mono-data-sm text-xs font-bold mb-1">
                <span>Berth Expansion</span>
                <span className="text-[10px] bg-cyan-950/90 text-cyan-300 border border-cyan-500/40 px-1.5 py-[2px] rounded font-mono-data-sm">+4.2 dB</span>
              </div>
              <p className="text-[11px] font-body-sm text-slate-300 leading-snug">SAR radar double-bounce: newly added gantry cranes.</p>
              <div className="mt-1 text-[10px] font-mono-data-sm text-cyan-300/90">STRUCTURE: HIGH DIELECTRIC METAL</div>
            </div>
          </div>
        </div>

        {/* Target Gamma */}
        <div className={`${getTargetStyle('targetGamma')} top-[68%] left-[64%] pointer-events-auto`}>
          <div className="relative">
            <span className="w-5 h-5 rounded-full bg-violet-400 border-2 border-slate-900 flex items-center justify-center text-[10px] font-bold text-slate-950 shadow-[0_0_10px_#a78bfa] font-mono-data-sm">03</span>
            <div className="absolute left-6 -top-3 w-48 p-2.5 rounded bg-[#070d18]/95 backdrop-blur border border-violet-500/50 shadow-2xl pointer-events-auto transition-transform group-hover:scale-105 opacity-0 invisible group-hover:opacity-100 group-hover:visible">
              <div className="flex items-center justify-between text-violet-300 font-mono-data-sm text-xs font-bold mb-1">
                <span>Fairway Channel</span>
                <span className="text-[10px] bg-violet-950/90 text-violet-300 border border-violet-500/40 px-1.5 py-[2px] rounded font-mono-data-sm">8 vessels</span>
              </div>
              <p className="text-[11px] font-body-sm text-slate-300 leading-snug">Verified acoustic and radar transit signatures.</p>
              <div className="mt-1 text-[10px] font-mono-data-sm text-violet-300/90">VESSEL CLASS: PANAMAX FREIGHT</div>
            </div>
          </div>
        </div>
        </div> {/* End Target Markers Layer */}

        {/* Swipe Slider (Fixed in Viewport) */}
        <div 
          className="absolute top-0 bottom-0 w-0.5 bg-cyan-400 shadow-[0_0_12px_#38bdf8] z-30" 
          style={{ left: `${sliderPos}%` }}
        >
          <div 
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-[#070e1b] border-2 border-cyan-400 text-cyan-300 flex items-center justify-center shadow-[0_0_12px_rgba(56,189,248,0.5)] pointer-events-auto hover:scale-110 transition-transform cursor-ew-resize"
            onMouseDown={(e) => { e.preventDefault(); e.stopPropagation(); setIsDraggingSlider(true); }}
            onTouchStart={(e) => { e.stopPropagation(); setIsDraggingSlider(true); }}
          >
            <span className="material-symbols-outlined text-base font-bold">swap_horiz</span>
          </div>
        </div>

        {/* GIS Data overlays (Fixed in Viewport) */}
        <div className="absolute top-2 left-20 z-10 pointer-events-none font-mono-data-sm text-[10px] text-cyan-400/70 bg-[#050b14]/80 px-2 py-0.5 rounded border border-slate-800/80">
          41°21'30"N | 02°09'15"E • GRID: UTM 43N
        </div>

        {/* Viewport Controls */}
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-[#070e1b]/95 backdrop-blur border border-slate-700/80 px-2 py-1 rounded flex items-center gap-1.5 shadow-2xl z-30">
          <button onClick={() => setScale(s => Math.min(s + 0.5, 4))} className="p-1.5 rounded hover:bg-slate-800 text-slate-300 hover:text-cyan-400 transition-colors" title="Zoom In">
            <span className="material-symbols-outlined text-base">zoom_in</span>
          </button>
          <button onClick={() => { setScale(s => Math.max(s - 0.5, 1)); if(scale - 0.5 <= 1) { setPanX(0); setPanY(0); } }} className="p-1.5 rounded hover:bg-slate-800 text-slate-300 hover:text-cyan-400 transition-colors" title="Zoom Out">
            <span className="material-symbols-outlined text-base">zoom_out</span>
          </button>
          <div className="h-4 w-px bg-slate-800"></div>
          <button onClick={() => alert('Measure tool activated. Click on the map to draw distances.')} className="p-1.5 rounded hover:bg-slate-800 text-slate-300 hover:text-cyan-400 transition-colors" title="Measure">
            <span className="material-symbols-outlined text-base">straighten</span>
          </button>
          <button onClick={toggleFullscreen} className="p-1.5 rounded hover:bg-slate-800 text-slate-300 hover:text-cyan-400 transition-colors" title={isFullscreen ? "Exit Fullscreen" : "Enter Fullscreen"}>
            <span className="material-symbols-outlined text-base">{isFullscreen ? 'fullscreen_exit' : 'fullscreen'}</span>
          </button>
        </div>
      </div>
    </section>
  );
};
