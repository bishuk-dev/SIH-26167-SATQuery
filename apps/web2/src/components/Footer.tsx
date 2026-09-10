
export const Footer = () => {
  return (
    <footer className="relative z-20 w-full border-t border-slate-800/90 bg-space-950/85 backdrop-blur-xl py-2 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-y-2 text-[11px] font-mono text-slate-400">
        {/* Left: Constellation Status */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 text-slate-200">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span className="font-medium">Orbital Sync: Active</span>
            <span className="text-slate-500">•</span>
            <span className="text-slate-300">14 Satellites Locked</span>
          </div>
          <span className="hidden md:inline text-slate-600">|</span>
          <div className="hidden md:flex items-center space-x-1 text-slate-300">
            <span className="text-slate-400">Resolution:</span>
            <span className="text-emerald-300">10m Optical / 5m SAR C-Band</span>
          </div>
        </div>
        {/* Right: Latency & Spatial Coordinates Reference */}
        <div className="flex items-center space-x-4 ml-auto">
          <div className="flex items-center space-x-1.5">
            <span className="text-slate-400">Latency:</span>
            <span className="text-slate-200">&lt; 800ms</span>
          </div>
          <span className="text-slate-600">|</span>
          <div className="flex items-center space-x-1.5 text-slate-300">
            <svg className="w-3 h-3 text-sky-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="10"></circle>
              <polygon points="12 8 8 12 12 16 16 12 12 8"></polygon>
            </svg>
            <span className="text-slate-200">N 41°21' E 02°09' • EPSG:32643</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
