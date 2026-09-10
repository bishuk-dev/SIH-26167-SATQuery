import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';

interface Message {
  id: string;
  role: 'user' | 'ai';
  content: string;
  isAnalyzing?: boolean;
}

export const AnalysisResponse = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const initialQuery = location.state?.query || '';
  
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (initialQuery && messages.length === 0) {
      handleQuery(initialQuery);
    } else if (!initialQuery && messages.length === 0) {
      navigate('/');
    }
  }, [initialQuery]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleQuery = async (queryText: string) => {
    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: queryText };
    const aiPlaceholderId = (Date.now() + 1).toString();
    
    setMessages(prev => [...prev, userMsg, { id: aiPlaceholderId, role: 'ai', content: '', isAnalyzing: true }]);
    setIsProcessing(true);
    setInput('');

    try {
      const res = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ observation_id: 'demo', question: queryText })
      });
      
      let aiResponseText = '';
      if (res.ok) {
        const data = await res.json();
        aiResponseText = data.answer || data.reasoning || JSON.stringify(data);
      } else {
        const errData = await res.json().catch(() => null);
        aiResponseText = `Error: ${errData?.error?.user_message || res.statusText}`;
      }

      setMessages(prev => prev.map(msg => 
        msg.id === aiPlaceholderId 
          ? { ...msg, content: aiResponseText, isAnalyzing: false } 
          : msg
      ));
    } catch (e: any) {
      setMessages(prev => prev.map(msg => 
        msg.id === aiPlaceholderId 
          ? { ...msg, content: `Failed to connect to AI engine: ${e.message}. Please ensure the backend is running.`, isAnalyzing: false } 
          : msg
      ));
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="relative z-10 w-full min-h-[calc(100vh-64px)] pt-20 pb-24 px-4 sm:px-6 flex flex-col items-center">
      
      {/* Header Bar */}
      <div className="w-full max-w-4xl mb-6 flex items-center justify-between bg-slate-900/60 backdrop-blur-xl border border-slate-800 rounded-xl p-4 shadow-lg">
        <div className="flex items-center gap-3">
          <button onClick={() => navigate('/')} className="p-1.5 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-white transition-colors">
            <span className="material-symbols-outlined text-[20px]">arrow_back</span>
          </button>
          <div className="h-4 w-px bg-slate-700"></div>
          <div className="flex flex-col">
            <h2 className="text-sm font-bold text-slate-200 tracking-wide">Analysis Session</h2>
            <div className="flex items-center gap-1.5 text-[10px] font-mono text-cyan-400">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse shadow-[0_0_8px_rgba(34,211,238,0.8)]"></span>
              RS-LLaVA ENGINE ONLINE
            </div>
          </div>
        </div>
        <div className="hidden sm:flex items-center gap-3">
           <button className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-xs font-mono text-slate-300 transition-colors flex items-center gap-1.5">
             <span className="material-symbols-outlined text-[14px]">download</span> Export Report
           </button>
        </div>
      </div>

      {/* Chat Area */}
      <div ref={scrollRef} className="w-full max-w-4xl flex-1 overflow-y-auto custom-scroll space-y-6 pb-6 pr-2">
        {messages.map((msg) => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] rounded-2xl p-5 ${
              msg.role === 'user' 
                ? 'bg-gradient-to-br from-fuchsia-900/40 to-purple-800/20 border border-fuchsia-500/30 text-slate-200' 
                : 'bg-[#11052C]/90 backdrop-blur-xl border border-cyan-500/30 text-slate-300 shadow-[0_0_30px_rgba(0,240,255,0.05)]'
            }`}>
              {msg.role === 'ai' && (
                <div className="flex items-center gap-2 mb-3 border-b border-cyan-900/50 pb-2">
                  <span className="material-symbols-outlined text-cyan-400 text-[18px]">psychology</span>
                  <span className="text-xs font-headline-sm font-bold text-cyan-400 tracking-wider">SATQUERY AI</span>
                  {msg.isAnalyzing && (
                     <span className="ml-auto flex items-center gap-1">
                       <span className="w-1 h-1 bg-cyan-400 rounded-full animate-bounce"></span>
                       <span className="w-1 h-1 bg-cyan-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></span>
                       <span className="w-1 h-1 bg-cyan-400 rounded-full animate-bounce" style={{animationDelay: '0.4s'}}></span>
                     </span>
                  )}
                </div>
              )}
              
              {msg.isAnalyzing ? (
                <div className="space-y-3 opacity-50">
                  <div className="h-4 bg-slate-700 rounded w-3/4 animate-pulse"></div>
                  <div className="h-4 bg-slate-700 rounded w-full animate-pulse"></div>
                  <div className="h-4 bg-slate-700 rounded w-5/6 animate-pulse"></div>
                </div>
              ) : (
                <div className="text-sm md:text-base font-sans leading-relaxed whitespace-pre-wrap prose prose-invert prose-cyan max-w-none">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
              )}

              {msg.role === 'ai' && !msg.isAnalyzing && msg.content && !msg.content.startsWith('Failed to connect') && (
                <div className="mt-4 pt-3 border-t border-cyan-900/30 flex items-center gap-4 text-[10px] font-mono text-cyan-500/70">
                  <div className="flex flex-col gap-0.5">
                    <span className="uppercase tracking-widest text-slate-500">Confidence</span>
                    <span className="text-fuchsia-400 font-bold">92%</span>
                  </div>
                  <div className="h-6 w-px bg-cyan-900/50"></div>
                  <div className="flex flex-col gap-0.5">
                    <span className="uppercase tracking-widest text-slate-500">Source</span>
                    <span className="text-cyan-300">Sentinel-2</span>
                  </div>
                  <div className="h-6 w-px bg-cyan-900/50"></div>
                  <button onClick={() => navigate('/workspace')} className="ml-auto px-3 py-1.5 border border-cyan-800/50 hover:border-fuchsia-500/50 bg-[#0d0221] hover:bg-[#1a053a] rounded transition-colors flex items-center gap-1.5 text-cyan-300">
                    <span className="material-symbols-outlined text-[14px] text-fuchsia-400">my_location</span>
                    Open in Workspace
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Input Area */}
      <div className="w-full max-w-4xl mt-auto pt-4 relative">
        <div className="relative bg-[#11052C]/90 backdrop-blur-xl border border-cyan-800/50 rounded-xl shadow-[0_0_40px_rgba(0,240,255,0.1)] p-2 flex items-end gap-2 focus-within:border-fuchsia-500/50 transition-colors">
          <button className="p-2 text-cyan-500/70 hover:text-fuchsia-400 transition-colors mb-1">
            <span className="material-symbols-outlined">add_circle</span>
          </button>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (input.trim() && !isProcessing) handleQuery(input);
              }
            }}
            placeholder="Ask a follow-up question..."
            className="flex-1 bg-transparent border-0 resize-none max-h-32 min-h-[44px] py-3 text-cyan-50 placeholder:text-cyan-700 text-sm focus:ring-0 custom-scroll font-sans"
            rows={1}
          />
          <button 
            onClick={() => { if (input.trim() && !isProcessing) handleQuery(input); }}
            disabled={!input.trim() || isProcessing}
            className={`p-2.5 rounded-lg mb-1 flex items-center justify-center transition-all ${
              input.trim() && !isProcessing 
                ? 'bg-fuchsia-600 hover:bg-fuchsia-500 text-white shadow-[0_0_20px_rgba(217,70,239,0.4)]' 
                : 'bg-[#1a053a] text-cyan-900'
            }`}
          >
            <span className="material-symbols-outlined text-[18px]">arrow_upward</span>
          </button>
        </div>
      </div>
      
    </div>
  );
};
