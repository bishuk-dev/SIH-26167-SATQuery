import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export const ChatInterface = () => {
  const [query, setQuery] = useState('');
  const [isDrawing, setIsDrawing] = useState(false);
  const [isSARSelected, setIsSARSelected] = useState(true);
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const handleRunAI = () => {
    if (!query.trim()) return;
    navigate('/analysis', { state: { query } });
  };

  const setPrompt = (text: string) => {
    setQuery(text);
  };

  const handleOptimizePrompt = () => {
    if (!query) {
      setQuery('Detect urban sprawl and lake boundary contraction in Bengaluru from 2022 to 2024 using multi-temporal Sentinel-2 imagery.');
    } else {
      setQuery((prev) => prev + ' with high-resolution radar backscatter analysis to reduce cloud-cover interference.');
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      alert(`File "${e.target.files[0].name}" selected for analysis.`);
    }
  };

  return (
    <main className="relative z-10 flex-1 flex flex-col items-center justify-center px-4 sm:px-6 lg:px-8 max-w-6xl mx-auto w-full pt-4 pb-16" data-purpose="hero-query-workspace">
      {/* Top Observation Category Pill */}
      <div className="mb-5 inline-flex items-center gap-2.5 px-4 py-1.5 rounded-full border border-purple-500/35 bg-purple-950/40 shadow-inner backdrop-blur-xl text-purple-200 text-xs font-semibold tracking-wider uppercase" data-purpose="status-badge">
        <span className="w-2.5 h-2.5 rounded-full bg-purple-400 shadow-sm shadow-purple-300 animate-pulse"></span>
        ORBITAL OBSERVATION & AI CHANGE DETECTION
      </div>

      {/* Main Dynamic Title */}
      <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-extrabold text-center tracking-tight leading-[1.1] max-w-4xl text-white drop-shadow-sm">
        Ask anything about our<br/>
        <span className="bg-gradient-to-r from-teal-300 via-cyan-400 to-purple-400 bg-clip-text text-transparent underline decoration-cyan-400/20 decoration-wavy decoration-1 underline-offset-8">
          changing planet.
        </span>
      </h1>

      {/* Descriptive Subtitle */}
      <p className="mt-4 sm:mt-5 text-sm sm:text-base md:text-lg text-slate-300/90 text-center max-w-2xl font-normal leading-relaxed">
        Natural language spatial queries powered by multi-temporal Sentinel-1/2, Landsat-9, and synthetic aperture radar.
      </p>

      {/* SearchConsole */}
      <div className="mt-8 w-full max-w-3xl" data-purpose="spatial-query-card">
        <div className="rounded-2xl border border-blue-500/30 bg-slate-950/65 backdrop-blur-2xl p-4 sm:p-5 shadow-2xl shadow-cyan-950/50 relative overflow-hidden group hover:border-blue-400/50 transition-colors">
          {/* Subtle Top Glow Line inside card */}
          <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-400/40 to-transparent"></div>
          
          {/* Query Input Box with AI icon */}
          <div className="relative flex items-start justify-between min-h-[96px] sm:min-h-[105px]">
            <div className="w-full pr-12">
              <label className="sr-only" htmlFor="query-input">Satellite AI Prompt Query</label>
              <textarea 
                className="w-full resize-none border-0 bg-transparent p-0 text-sm sm:text-base text-slate-100 placeholder-slate-400/80 focus:ring-0 focus:outline-none leading-relaxed font-normal custom-scroll" 
                id="query-input" 
                placeholder="Describe an area of interest or change detection query (e.g. Detect urban sprawl and lake boundary contraction in Bengaluru from 2022 to 2024)..." 
                rows={3}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
            
            {/* Magic Sparkle Action Indicator */}
            <div className="absolute top-0 right-0">
              <button 
                onClick={handleOptimizePrompt}
                aria-label="Optimize with AI" 
                className="p-2.5 rounded-xl bg-blue-900/30 hover:bg-blue-800/50 border border-blue-500/30 text-cyan-400 hover:text-cyan-300 transition-all flex items-center justify-center group" 
                title="Optimize prompt with AI"
                type="button"
              >
                <svg className="w-5 h-5 transition-transform group-hover:scale-110 group-active:rotate-12" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 2l2.4 6.6L21 11l-6.6 2.4L12 20l-2.4-6.6L3 11l6.6-2.4L12 2z"></path>
                </svg>
              </button>
            </div>
          </div>
          
          {/* Bottom Controls & CTA */}
          <div className="mt-4 pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2.5" data-purpose="query-controls">
            {/* Filter / Tool Attachment Chips */}
            <div className="flex flex-wrap items-center gap-2">
              <input 
                type="file" 
                ref={fileInputRef} 
                className="hidden" 
                accept=".tif,.tiff,image/tiff"
                onChange={handleFileUpload}
              />
              <button 
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs sm:text-sm font-medium bg-slate-900/90 hover:bg-slate-800 text-slate-200 border border-slate-700/70 hover:border-slate-600 transition-colors" 
                type="button"
              >
                <svg className="w-4 h-4 text-slate-300" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="17 8 12 3 7 8"></polyline>
                  <line x1="12" x2="12" y1="3" y2="15"></line>
                </svg>
                Add Imagery (GeoTIFF)
              </button>
              <button 
                onClick={() => setIsDrawing(!isDrawing)}
                className={`inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs sm:text-sm font-medium transition-colors border ${
                  isDrawing ? 'bg-cyan-900/50 text-cyan-300 border-cyan-500/50' : 'bg-slate-900/50 text-slate-200 border-slate-700/60 hover:bg-slate-800 hover:border-slate-500'
                }`} 
                type="button"
              >
                <svg className={`w-4 h-4 ${isDrawing ? 'text-cyan-300' : 'text-cyan-400'}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <path d="M3 3h6v2H5v4H3V3z"></path>
                  <path d="M21 3h-6v2h4v4h2V3z"></path>
                  <path d="M3 21h6v-2H5v-4H3v6z"></path>
                  <path d="M21 21h-6v-2h4v-4h2v6z"></path>
                </svg>
                {isDrawing ? 'Drawing Box...' : 'Bounding Box'}
              </button>
              <button 
                onClick={() => setIsSARSelected(!isSARSelected)}
                className={`inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs sm:text-sm font-medium border transition-colors ${
                  isSARSelected ? 'bg-slate-900/40 text-amber-300/95 border-amber-500/30' : 'bg-slate-900/40 text-slate-400 border-slate-700/60 hover:bg-slate-800/80'
                }`}
                type="button"
              >
                <svg className={`w-4 h-4 ${isSARSelected ? 'text-amber-400' : 'text-slate-500'}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <path d="M4.9 19.1C1 15.2 1 8.8 4.9 4.9"></path>
                  <path d="M7.8 16.2c-2.3-2.3-2.3-6.1 0-8.5"></path>
                  <circle cx="12" cy="12" r="2"></circle>
                  <path d="M16.2 7.8c2.3 2.3 2.3 6.1 0 8.5"></path>
                  <path d="M19.1 4.9C23 8.8 23 15.2 19.1 19.1"></path>
                </svg>
                {isSARSelected ? 'Sentinel-2 & SAR' : 'Sentinel-2 Only'}
              </button>
            </div>
            
            {/* Primary "Query Orbit" Action CTA */}
            <button 
              onClick={handleRunAI}
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-2.5 rounded-xl font-semibold text-sm text-white bg-gradient-to-r from-purple-600 via-indigo-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 shadow-lg shadow-indigo-600/35 active:scale-[0.98] transition-all duration-150" 
              type="button"
            >
              <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.2" viewBox="0 0 24 24">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" x2="16.65" y1="21" y2="16.65"></line>
              </svg>
              Query Orbit
            </button>
          </div>
        </div>
      </div>

      {/* SuggestionPills */}
      <div className="mt-6 flex flex-wrap items-center justify-center gap-2 max-w-3xl text-xs sm:text-sm" data-purpose="prompt-suggestions">
        <span className="text-slate-400 font-medium mr-1">Try asking:</span>
        <button className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border border-slate-700/70 bg-slate-900/50 hover:bg-slate-800/80 hover:border-cyan-500/40 text-slate-300 hover:text-white transition-all backdrop-blur-md" onClick={() => setPrompt('Port terminal expansion in Barcelona')} type="button">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
          Port terminal expansion in Barcelona
        </button>
        <button className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border border-slate-700/70 bg-slate-900/50 hover:bg-slate-800/80 hover:border-cyan-500/40 text-slate-300 hover:text-white transition-all backdrop-blur-md" onClick={() => setPrompt('Lithium brine evaporation delta')} type="button">
          <span className="w-1.5 h-1.5 rounded-full bg-blue-400"></span>
          Lithium brine evaporation delta
        </button>
        <button className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border border-slate-700/70 bg-slate-900/50 hover:bg-slate-800/80 hover:border-cyan-500/40 text-slate-300 hover:text-white transition-all backdrop-blur-md" onClick={() => setPrompt('Amazon canopy loss Q1 2024')} type="button">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
          Amazon canopy loss Q1 2024
        </button>
        <button className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border border-slate-700/70 bg-slate-900/50 hover:bg-slate-800/80 hover:border-cyan-500/40 text-slate-300 hover:text-white transition-all backdrop-blur-md" onClick={() => setPrompt('Rotterdam tanker fairway count')} type="button">
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-400"></span>
          Rotterdam tanker fairway count
        </button>
      </div>
    </main>
  );
};
