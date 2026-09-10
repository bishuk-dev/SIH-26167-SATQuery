import React, { useState } from 'react';

interface LeftPanelProps {
  onInspect: (targetId: string) => void;
  observationId: string;
}

export const LeftPanel: React.FC<LeftPanelProps> = ({ onInspect, observationId }) => {
  const [query, setQuery] = useState('Detect urban expansion around Bengaluru between 2022 and 2024 and highlight the changed areas.');
  const [isExecuting, setIsExecuting] = useState(false);
  const [hasResult, setHasResult] = useState(true); // Mocking that a result is already there for the demo
  const [summaryText, setSummaryText] = useState('Urban expansion increased noticeably along the northern boundary of the study area. Major backscatter variance detected in the newly constructed residential sectors.');
  const [showEvidence, setShowEvidence] = useState(false);

  const handleExecute = async () => {
    if (!query) return;
    setIsExecuting(true);
    setHasResult(false);
    setSummaryText('Analyzing orbital data...');
    try {
      const res = await fetch('http://localhost:8000/api/vqa', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ observation_id: observationId, question: query })
      });
      if (res.ok) {
        const data = await res.json();
        setSummaryText(data.answer || data.reasoning || JSON.stringify(data));
        setHasResult(true);
      } else {
        const errData = await res.json();
        setSummaryText(`Error: ${errData.error?.user_message || res.statusText}`);
      }
    } catch (e: any) {
      setSummaryText(`Failed to connect to backend: ${e.message}`);
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <aside className="w-[380px] border-r border-slate-800 bg-[#0a101d] text-slate-200 flex flex-col z-20 shrink-0 shadow-2xl relative">
      
      {/* Top Section: AI QUERY */}
      <div className="flex-1 flex flex-col min-h-[50%] border-b border-slate-800">
        <div className="p-4 border-b border-slate-800/60 bg-[#05080f]/50 flex items-center gap-2">
          <span className="material-symbols-outlined text-emerald-400 text-lg">psychology</span>
          <h2 className="font-sans font-bold text-sm tracking-widest text-slate-200">AI QUERY</h2>
        </div>
        
        <div className="p-4 flex-1 flex flex-col gap-4">
          <div>
            <label className="text-[10px] font-mono text-slate-500 uppercase tracking-widest mb-2 block">Ask your question</label>
            <textarea 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full h-28 bg-slate-900/50 border border-slate-700/50 rounded-lg p-3 text-sm text-slate-200 font-sans focus:outline-none focus:border-emerald-500/50 focus:bg-slate-900 transition-colors resize-none placeholder-slate-600"
              placeholder="e.g. Detect urban expansion..."
            />
          </div>

          <div className="flex items-center justify-between mt-auto">
            <div className="flex flex-col">
              <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">Model</span>
              <span className="text-xs font-mono text-slate-300">RS-LLaVA</span>
            </div>
            <button 
              onClick={handleExecute}
              disabled={isExecuting}
              className={`px-6 py-2 rounded border font-sans text-xs font-bold tracking-wide transition-all flex items-center gap-2
                ${isExecuting ? 'bg-slate-800 border-slate-700 text-slate-400' : 'bg-emerald-600/10 hover:bg-emerald-600 border-emerald-500/50 text-emerald-400 hover:text-white shadow-[0_0_10px_rgba(16,185,129,0.1)] hover:shadow-[0_0_15px_rgba(16,185,129,0.4)]'}`}
            >
              {isExecuting ? (
                <><span className="material-symbols-outlined text-[16px] animate-spin">refresh</span> Processing</>
              ) : (
                <>Run AI <span className="material-symbols-outlined text-[16px]">arrow_forward</span></>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Bottom Section: AI ANALYSIS */}
      <div className="flex-1 flex flex-col min-h-[50%] bg-[#080d18] overflow-y-auto custom-scrollbar">
        <div className="p-4 border-b border-slate-800/60 flex items-center gap-2 sticky top-0 bg-[#080d18]/90 backdrop-blur-sm z-10">
          <span className="material-symbols-outlined text-emerald-400 text-lg">analytics</span>
          <h2 className="font-sans font-bold text-sm tracking-widest text-slate-200">AI ANALYSIS</h2>
        </div>

        {hasResult ? (
          <div className="p-5 flex flex-col gap-6">
            <div>
              <h3 className="text-[10px] font-mono text-slate-500 uppercase tracking-widest mb-1.5">Question</h3>
              <p className="text-sm text-slate-300 font-sans italic border-l-2 border-slate-700 pl-3 py-0.5">{query}</p>
            </div>

            <div>
              <h3 className="text-[10px] font-mono text-slate-500 uppercase tracking-widest mb-2">Result</h3>
              <p className="text-sm text-slate-200 font-sans leading-relaxed">
                {summaryText}
              </p>
            </div>

            <div className="grid grid-cols-3 gap-3 border-y border-slate-800/60 py-3">
              <div className="flex flex-col gap-1">
                <span className="text-[9px] font-mono text-slate-500 uppercase tracking-widest">Confidence</span>
                <span className="text-sm font-mono font-bold text-emerald-400">92%</span>
              </div>
              <div className="flex flex-col gap-1 border-l border-slate-800/60 pl-3">
                <span className="text-[9px] font-mono text-slate-500 uppercase tracking-widest">Model</span>
                <span className="text-xs font-mono text-slate-300 mt-0.5">RS-LLaVA</span>
              </div>
              <div className="flex flex-col gap-1 border-l border-slate-800/60 pl-3">
                <span className="text-[9px] font-mono text-slate-500 uppercase tracking-widest">Source</span>
                <span className="text-xs font-mono text-slate-300 mt-0.5">Sentinel-2</span>
              </div>
            </div>

            <div className="flex flex-col gap-2 mt-2">
              <button onClick={() => onInspect('targetAlpha')} className="w-full py-2 bg-slate-800/50 hover:bg-slate-800 text-slate-300 border border-slate-700 hover:border-slate-500 rounded font-sans text-xs font-medium transition-colors flex items-center justify-center gap-2">
                <span className="material-symbols-outlined text-[16px] text-emerald-400">my_location</span>
                Show on Map
              </button>
              
              <div className="flex gap-2">
                <button onClick={() => setShowEvidence(!showEvidence)} className="flex-1 py-2 bg-slate-800/30 hover:bg-slate-800 text-slate-300 border border-slate-700/50 hover:border-slate-600 rounded font-sans text-xs font-medium transition-colors flex items-center justify-center gap-1.5">
                  <span className="material-symbols-outlined text-[16px] text-slate-400">visibility</span>
                  {showEvidence ? 'Hide Evidence' : 'View Evidence'}
                </button>
                <button className="flex-1 py-2 bg-slate-800/30 hover:bg-slate-800 text-slate-300 border border-slate-700/50 hover:border-slate-600 rounded font-sans text-xs font-medium transition-colors flex items-center justify-center gap-1.5">
                  <span className="material-symbols-outlined text-[16px] text-slate-400">download</span>
                  Export Report
                </button>
              </div>

              {showEvidence && (
                <div className="mt-3 p-3 bg-slate-900/50 border border-slate-800 rounded text-xs font-mono text-slate-400 space-y-2 animate-in fade-in slide-in-from-top-2 duration-200">
                  <p className="text-emerald-400/80 mb-1 tracking-widest uppercase text-[10px]">Primary Evidence Markers:</p>
                  <ul className="list-disc pl-4 space-y-1">
                    <li>14.6 hectares of new built-up area</li>
                    <li>Strongest change detected in north-west region</li>
                    <li>High variance in SAR dual-bounce signals</li>
                  </ul>
                </div>
              )}
            </div>

          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-slate-600 p-8 text-center gap-3">
            <span className="material-symbols-outlined text-4xl opacity-50">troubleshoot</span>
            <p className="text-sm font-sans">Run a query to generate an AI analysis report.</p>
          </div>
        )}
      </div>
    </aside>
  );
};
