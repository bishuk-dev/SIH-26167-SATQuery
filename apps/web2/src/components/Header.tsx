import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

export const Header = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [shareText, setShareText] = useState('Share');

  useEffect(() => {
    const handleScroll = () => setIsScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleShare = () => {
    navigator.clipboard.writeText(window.location.href);
    setShareText('Copied!');
    setTimeout(() => setShareText('Share'), 2000);
  };

  return (
    <header className={`relative z-20 w-full px-6 lg:px-12 py-5 flex items-center justify-between transition-all duration-300 ${isScrolled ? 'bg-[#030712]/80 backdrop-blur-xl border-b border-slate-800/50' : 'bg-transparent'}`} data-purpose="top-navigation-bar">
      {/* Left: Brand Logo & PRO badge */}
      <div className="flex items-center gap-3">
        <Link className="flex items-center gap-2.5 group" to="/">
          {/* Planetary Orbital Icon */}
          <div className="relative w-8 h-8 rounded-full bg-cyan-500/10 border border-cyan-400/30 flex items-center justify-center text-cyan-400 group-hover:scale-105 transition-transform">
            <svg className="w-5 h-5 text-cyan-400" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="6"></circle>
              <path d="M2.5 12c0-3.5 9-7 19-3"></path>
              <path d="M21.5 12c0 3.5-9 7-19 3"></path>
            </svg>
            <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
          </div>
          <span className="text-xl font-bold tracking-tight text-white flex items-baseline">
            SatQuery<span className="text-cyan-400 font-extrabold">.AI</span>
          </span>
        </Link>
        <span className="ml-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold tracking-wide uppercase bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 shadow-sm shadow-emerald-500/20">
          PRO
        </span>
      </div>

      {/* Center: Navigation Links */}
      <nav className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-300" data-purpose="desktop-nav-menu">
        <Link className="hover:text-white transition-colors duration-150" to="/docs">Analysis Docs</Link>
        <Link className="hover:text-white transition-colors duration-150 flex items-center gap-2" to="/catalog">
          Sensor Catalog
          <span className="w-2 h-2 rounded-full bg-teal-400 shadow-sm shadow-teal-300"></span>
        </Link>
        <Link className="hover:text-white transition-colors duration-150" to="/api-keys">API Keys</Link>
        <Link className="hover:text-white transition-colors duration-150" to="/benchmarks">Benchmarks</Link>
        <Link className="hover:text-white transition-colors duration-150" to="/docs">Docs</Link>
      </nav>

      {/* Right: Quick Actions */}
      <div className="flex items-center gap-3.5" data-purpose="navigation-actions">
        <button 
          onClick={handleShare}
          className="hidden sm:inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-slate-300 hover:text-white bg-slate-900/60 hover:bg-slate-800/80 border border-slate-700/60 rounded-xl backdrop-blur-md transition-all w-24 justify-center" 
          type="button"
        >
          {shareText === 'Share' ? (
            <svg className="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24">
              <circle cx="18" cy="5" r="3"></circle>
              <circle cx="6" cy="12" r="3"></circle>
              <circle cx="18" cy="19" r="3"></circle>
              <line x1="8.59" x2="15.42" y1="13.51" y2="17.49"></line>
              <line x1="15.41" x2="8.59" y1="6.51" y2="10.49"></line>
            </svg>
          ) : (
            <svg className="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24">
              <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
          )}
          <span>{shareText}</span>
        </button>
        <Link className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-bold text-xs sm:text-sm tracking-wide bg-gradient-to-r from-cyan-400 via-teal-400 to-emerald-400 text-slate-950 shadow-lg shadow-cyan-500/25 hover:shadow-cyan-400/40 hover:brightness-105 active:scale-[0.98] transition-all duration-200" to="/workspace">
          <svg className="w-4 h-4 text-slate-950" fill="currentColor" viewBox="0 0 24 24">
            <path d="M13.13 2.55a1.25 1.25 0 0 0-1.85.28L8.6 6.88a1.25 1.25 0 0 0 .1 1.63l.89.89-6.3 6.3a1.25 1.25 0 0 0 0 1.77l3.73 3.73a1.25 1.25 0 0 0 1.77 0l6.3-6.3.89.89a1.25 1.25 0 0 0 1.63.1l4.05-2.68a1.25 1.25 0 0 0 .28-1.85L13.13 2.55zM6.51 18.9l-2.6-2.6 4.96-4.96 2.6 2.6-4.96 4.96z"></path>
          </svg>
          Launch Workspace
        </Link>
      </div>
    </header>
  );
};
