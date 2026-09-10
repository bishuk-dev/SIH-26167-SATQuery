import React from 'react';

export const HomeFooter = () => {
  return (
    <footer className="relative z-20 w-full border-t border-slate-800/70 bg-[#030712]/80 backdrop-blur-xl py-6 px-6 lg:px-12 mt-auto" data-purpose="feature-highlights-bar">
      <div className="max-w-7xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-6 lg:gap-8">
        {/* Feature 1: Multi-Sensor Data */}
        <div className="flex items-center gap-3.5 group">
          <div className="w-11 h-11 rounded-xl bg-teal-500/10 border border-teal-500/20 flex items-center justify-center text-teal-400 group-hover:border-teal-400/50 transition-colors shadow-sm shadow-teal-500/10">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path d="M12 2a10 10 0 0 1 10 10"></path>
              <path d="M12 6a6 6 0 0 1 6 6"></path>
              <path d="M12 10a2 2 0 0 1 2 2"></path>
              <path d="M4 14l8-8"></path>
              <circle cx="4" cy="14" r="2"></circle>
            </svg>
          </div>
          <div>
            <h4 className="text-sm font-semibold text-white tracking-tight">Multi-Sensor Data</h4>
            <p className="text-xs text-slate-400">Sentinel-1/2, Landsat-9 & more</p>
          </div>
        </div>
        
        {/* Feature 2: AI-Powered Analysis */}
        <div className="flex items-center gap-3.5 group">
          <div className="w-11 h-11 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400 group-hover:border-purple-400/50 transition-colors shadow-sm shadow-purple-500/10">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <rect height="16" rx="2" width="16" x="4" y="4"></rect>
              <rect height="6" width="6" x="9" y="9"></rect>
              <line x1="9" x2="9" y1="1" y2="4"></line>
              <line x1="15" x2="15" y1="1" y2="4"></line>
              <line x1="9" x2="9" y1="20" y2="23"></line>
              <line x1="15" x2="15" y1="20" y2="23"></line>
              <line x1="20" x2="23" y1="9" y2="9"></line>
              <line x1="20" x2="23" y1="15" y2="15"></line>
              <line x1="1" x2="4" y1="9" y2="9"></line>
              <line x1="1" x2="4" y1="15" y2="15"></line>
            </svg>
          </div>
          <div>
            <h4 className="text-sm font-semibold text-white tracking-tight">AI-Powered Analysis</h4>
            <p className="text-xs text-slate-400">Fast, accurate, reliable</p>
          </div>
        </div>

        {/* Feature 3: Global Coverage */}
        <div className="flex items-center gap-3.5 group">
          <div className="w-11 h-11 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400 group-hover:border-blue-400/50 transition-colors shadow-sm shadow-blue-500/10">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="2" x2="22" y1="12" y2="12"></line>
              <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
            </svg>
          </div>
          <div>
            <h4 className="text-sm font-semibold text-white tracking-tight">Global Coverage</h4>
            <p className="text-xs text-slate-400">Land, ocean & atmosphere</p>
          </div>
        </div>

        {/* Feature 4: Real-Time Insights */}
        <div className="flex items-center gap-3.5 group">
          <div className="w-11 h-11 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 group-hover:border-emerald-400/50 transition-colors shadow-sm shadow-emerald-500/10">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
            </svg>
          </div>
          <div>
            <h4 className="text-sm font-semibold text-white tracking-tight">Real-Time Insights</h4>
            <p className="text-xs text-slate-400">For a sustainable tomorrow</p>
          </div>
        </div>
      </div>
    </footer>
  );
};
