import React from 'react';
import { useNavigate } from 'react-router-dom';

interface PageShellProps {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}

export const PageShell = ({ title, subtitle, children }: PageShellProps) => {
  const navigate = useNavigate();
  return (
    // fixed + z-50 ensures it sits above ThreeBackground / MapViewer in App.tsx
    <div
      className="fixed inset-0 z-50 overflow-y-auto flex flex-col"
      style={{
        backgroundColor: '#050d1e',
        fontFamily: "'Plus Jakarta Sans', sans-serif",
        WebkitFontSmoothing: 'antialiased',
        background: 'linear-gradient(135deg, #020b18 0%, #060d22 40%, #0a0f2e 100%)',
      }}
    >
      {/* Subtle grid texture */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage:
            'linear-gradient(rgba(6,182,212,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(6,182,212,0.03) 1px, transparent 1px)',
          backgroundSize: '40px 40px',
        }}
      />

      {/* Glow blobs */}
      <div className="fixed top-0 left-1/4 w-96 h-96 rounded-full bg-cyan-500/5 blur-[120px] pointer-events-none" />
      <div className="fixed bottom-0 right-1/4 w-96 h-96 rounded-full bg-indigo-500/5 blur-[120px] pointer-events-none" />

      {/* ── Top bar ── */}
      <header
        className="w-full px-6 lg:px-12 py-4 flex items-center gap-4 sticky top-0 z-10 border-b"
        style={{
          backgroundColor: 'rgba(5,13,30,0.9)',
          backdropFilter: 'blur(20px)',
          borderColor: 'rgba(255,255,255,0.06)',
        }}
      >
        <button
          onClick={() => navigate('/')}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-white/[0.07] border border-white/10 hover:border-white/20 text-xs font-semibold transition-all cursor-pointer"
          type="button"
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" viewBox="0 0 24 24">
            <polyline points="15 18 9 12 15 6" />
          </svg>
          Back to Home
        </button>
        <div className="h-5 w-px bg-white/10" />
        <div>
          <p className="text-[15px] font-bold text-white tracking-[-0.02em]">{title}</p>
          {subtitle && <p className="text-[11px] text-slate-500 mt-0.5">{subtitle}</p>}
        </div>
        <div className="ml-auto flex items-center gap-2">
          <svg width="24" height="24" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <linearGradient id="pgBg" x1="0" y1="0" x2="36" y2="36" gradientUnits="userSpaceOnUse">
                <stop offset="0%" stopColor="#0a1628" /><stop offset="100%" stopColor="#0d2240" />
              </linearGradient>
              <linearGradient id="pgOrbit" x1="0" y1="0" x2="36" y2="36" gradientUnits="userSpaceOnUse">
                <stop offset="0%" stopColor="#22d3ee" /><stop offset="100%" stopColor="#6366f1" />
              </linearGradient>
              <linearGradient id="pgEarth" x1="10" y1="10" x2="26" y2="26" gradientUnits="userSpaceOnUse">
                <stop offset="0%" stopColor="#0ea5e9" /><stop offset="100%" stopColor="#0284c7" />
              </linearGradient>
            </defs>
            <circle cx="18" cy="18" r="18" fill="url(#pgBg)" />
            <circle cx="18" cy="18" r="17" stroke="url(#pgOrbit)" strokeWidth="0.6" strokeOpacity="0.5" />
            <ellipse cx="18" cy="18" rx="13" ry="5.5" stroke="url(#pgOrbit)" strokeWidth="1" strokeDasharray="3 2" transform="rotate(-35 18 18)" />
            <circle cx="18" cy="18" r="5.5" fill="url(#pgEarth)" />
            <circle cx="27.5" cy="13.5" r="1.8" fill="#22d3ee" />
            <circle cx="27.5" cy="13.5" r="1" fill="white" />
          </svg>
          <span className="text-sm font-bold text-white tracking-[-0.03em]">
            SatQuery<span className="text-cyan-400">.AI</span>
          </span>
        </div>
      </header>

      {/* ── Page content ── */}
      <main className="relative z-[1] flex-1 px-4 sm:px-8 lg:px-16 py-10">
        {children}
      </main>
    </div>
  );
};
