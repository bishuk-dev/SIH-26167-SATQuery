import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import * as THREE from 'three';
import ReactMarkdown from 'react-markdown';

export function NewLandingPage() {
  const canvasRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const queryRef = useRef<HTMLTextAreaElement>(null);
  
  const [aiResponse, setAiResponse] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const setQuery = (text: string) => {
    if (queryRef.current) {
      queryRef.current.value = text;
      queryRef.current.focus();
    }
  };

  const handleQueryOrbit = async () => {
    const q = queryRef.current?.value.trim() || '';
    if (!q) return;

    setIsAnalyzing(true);
    setAiResponse(null);

    try {
      const res = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q })
      });
      
      let aiResponseText = '';
      if (res.ok) {
        const data = await res.json();
        aiResponseText = data.answer || data.reasoning || JSON.stringify(data);
      } else {
        const errData = await res.json().catch(() => null);
        aiResponseText = `Error: ${errData?.error?.user_message || res.statusText}`;
      }
      setAiResponse(aiResponseText);
    } catch (e: any) {
      setAiResponse(`Failed to connect to AI engine: ${e.message}. Please ensure the backend is running.`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ── Toolbar state ─────────────────────────────────────────────────────
  const [showBBox, setShowBBox] = useState(false);
  const [showSensor, setShowSensor] = useState(false);
  const [activeSensors, setActiveSensors] = useState<string[]>(['Sentinel-2', 'SAR']);
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  const [bbox, setBbox] = useState({ north: '', south: '', east: '', west: '' });
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFile(file.name);
      // Append file name to query
      if (queryRef.current) {
        const existing = queryRef.current.value.trim();
        queryRef.current.value = existing ? `${existing} [File: ${file.name}]` : `Analyze: ${file.name}`;
        queryRef.current.focus();
      }
    }
  };

  const toggleSensor = (s: string) => {
    setActiveSensors(prev => prev.includes(s) ? prev.filter(x => x !== s) : [...prev, s]);
  };

  const applyBBox = () => {
    const { north, south, east, west } = bbox;
    if (north && south && east && west && queryRef.current) {
      const existing = queryRef.current.value.trim();
      const bboxStr = `[BBox: N${north} S${south} E${east} W${west}]`;
      queryRef.current.value = existing ? `${existing} ${bboxStr}` : bboxStr;
      queryRef.current.focus();
    }
    setShowBBox(false);
  };

  useEffect(() => {
    const container = canvasRef.current;
    if (!container) return;

    // ── Scene Setup ──────────────────────────────────────────────────────
    const w = window.innerWidth;
    const h = window.innerHeight;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 1000);
    camera.position.set(0.45, 0.05, 1.55);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h);
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    // ── Lights ───────────────────────────────────────────────────────────
    scene.add(new THREE.AmbientLight(0x111133, 2.0));
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(5, 3, 5);
    scene.add(sun);
    const rim = new THREE.DirectionalLight(0x06b6d4, 0.8);
    rim.position.set(-4, -2, -3);
    scene.add(rim);

    // ── Starfield ────────────────────────────────────────────────────────
    const starsGeo = new THREE.BufferGeometry();
    const starCount = 2000;
    const starPos = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount * 3; i++) {
      starPos[i] = (Math.random() - 0.5) * 180;
    }
    starsGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
    const starsMat = new THREE.PointsMaterial({
      color: 0xffffff, size: 0.15, transparent: true, opacity: 0.75,
      blending: THREE.AdditiveBlending,
    });
    scene.add(new THREE.Points(starsGeo, starsMat));

    // ── Earth ─────────────────────────────────────────────────────────────
    const loader = new THREE.TextureLoader();
    const radius = 0.5;
    const seg = 64;

    const earthMat = new THREE.MeshPhongMaterial({
      map: loader.load('/webgl-earth-images/2_no_clouds_4k.jpg'),
      bumpMap: loader.load('/webgl-earth-images/elev_bump_4k.jpg'),
      bumpScale: 0.005,
      specularMap: loader.load('/webgl-earth-images/water_4k.png'),
      specular: new THREE.Color(0x1a3a5c),
      shininess: 18,
    });
    const earth = new THREE.Mesh(new THREE.SphereGeometry(radius, seg, seg), earthMat);
    earth.rotation.y = 6;
    scene.add(earth);

    const cloudMat = new THREE.MeshPhongMaterial({
      map: loader.load('/webgl-earth-images/fair_clouds_4k.png'),
      transparent: true, opacity: 0.82,
    });
    const clouds = new THREE.Mesh(new THREE.SphereGeometry(radius + 0.003, seg, seg), cloudMat);
    clouds.rotation.y = 6;
    scene.add(clouds);

    // Atmosphere glow
    const atmoMat = new THREE.MeshBasicMaterial({
      color: 0x0284c7, transparent: true, opacity: 0.13,
      blending: THREE.AdditiveBlending, side: THREE.BackSide,
    });
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(radius * 1.08, 48, 48), atmoMat));

    // ── Orbital ring ─────────────────────────────────────────────────────
    const ringPts: THREE.Vector3[] = [];
    for (let i = 0; i <= 128; i++) {
      const a = (i / 128) * Math.PI * 2;
      ringPts.push(new THREE.Vector3(Math.cos(a) * 0.72, 0, Math.sin(a) * 0.72));
    }
    const ring = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(ringPts),
      new THREE.LineBasicMaterial({ color: 0x06b6d4, transparent: true, opacity: 0.35 }),
    );
    ring.rotation.x = 0.45;
    scene.add(ring);

    // ── Satellite ────────────────────────────────────────────────────────
    const satGroup = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(0.04, 0.04, 0.08),
      new THREE.MeshPhongMaterial({ color: 0xd97706 }),
    );
    satGroup.add(body);
    const panelMat = new THREE.MeshPhongMaterial({ color: 0x1e40af, emissive: 0x172554, emissiveIntensity: 0.4 });
    [-1, 1].forEach(side => {
      const panel = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.002, 0.055), panelMat);
      panel.position.set(side * 0.11, 0, 0);
      satGroup.add(panel);
    });
    scene.add(satGroup);

    // ── Animation ────────────────────────────────────────────────────────
    let animId: number;
    let theta = 0;

    const animate = () => {
      animId = requestAnimationFrame(animate);
      earth.rotation.y += 0.0008;
      clouds.rotation.y += 0.001;
      theta += 0.004;
      satGroup.position.set(
        Math.cos(theta) * 0.72,
        Math.sin(theta * 0.6) * 0.15,
        Math.sin(theta) * 0.72,
      );
      satGroup.lookAt(0, 0, 0);
      renderer.render(scene, camera);
    };
    animate();

    // ── Resize ───────────────────────────────────────────────────────────
    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener('resize', onResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', onResize);
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, []);

  return (
    <div
      className="relative min-h-screen text-slate-100 flex flex-col justify-between select-none overflow-hidden"
      style={{ backgroundColor: '#000', fontFamily: "'Plus Jakarta Sans', sans-serif", WebkitFontSmoothing: 'antialiased' }}
    >
      {/* ── Three.js Canvas Background ── */}
      <div
        ref={canvasRef}
        className="fixed inset-0 w-full h-full pointer-events-none z-0"
      />

      {/* ── Vignette overlay ── */}
      <div
        className="fixed inset-0 pointer-events-none z-[1]"
        style={{
          background:
            'radial-gradient(ellipse at 72% 52%, transparent 28%, rgba(0,0,0,0.6) 75%),' +
            'linear-gradient(to right, rgba(0,0,0,0.82) 0%, rgba(0,0,0,0.2) 50%, transparent 100%)',
        }}
      />

      {/* ════ HEADER ════ */}
      <header className="relative z-20 w-full px-6 lg:px-12 py-5 flex items-center justify-between pointer-events-auto">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <a className="flex items-center gap-2.5 group" href="#">
            {/* Professional logo mark */}
            <div className="relative w-9 h-9 group-hover:scale-105 transition-transform duration-200">
              <svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
                <defs>
                  <linearGradient id="bgGrad" x1="0" y1="0" x2="36" y2="36" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stopColor="#0a1628" />
                    <stop offset="100%" stopColor="#0d2240" />
                  </linearGradient>
                  <linearGradient id="orbitGrad" x1="0" y1="0" x2="36" y2="36" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stopColor="#22d3ee" />
                    <stop offset="100%" stopColor="#6366f1" />
                  </linearGradient>
                  <linearGradient id="earthGrad" x1="10" y1="10" x2="26" y2="26" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stopColor="#0ea5e9" />
                    <stop offset="100%" stopColor="#0284c7" />
                  </linearGradient>
                  <radialGradient id="glowGrad" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.15" />
                    <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
                  </radialGradient>
                  <filter id="glow">
                    <feGaussianBlur stdDeviation="1.2" result="coloredBlur" />
                    <feMerge><feMergeNode in="coloredBlur" /><feMergeNode in="SourceGraphic" /></feMerge>
                  </filter>
                </defs>
                {/* Background circle */}
                <circle cx="18" cy="18" r="18" fill="url(#bgGrad)" />
                {/* Glow fill */}
                <circle cx="18" cy="18" r="18" fill="url(#glowGrad)" />
                {/* Outer border ring */}
                <circle cx="18" cy="18" r="17" stroke="url(#orbitGrad)" strokeWidth="0.6" strokeOpacity="0.5" />
                {/* Orbit ellipse - tilted */}
                <ellipse cx="18" cy="18" rx="13" ry="5.5" stroke="url(#orbitGrad)" strokeWidth="1" strokeDasharray="3 2" transform="rotate(-35 18 18)" filter="url(#glow)" />
                {/* Earth sphere */}
                <circle cx="18" cy="18" r="5.5" fill="url(#earthGrad)" />
                {/* Earth highlight */}
                <circle cx="16" cy="16" r="1.8" fill="white" fillOpacity="0.12" />
                {/* Satellite dot on orbit */}
                <circle cx="27.5" cy="13.5" r="1.8" fill="#22d3ee" filter="url(#glow)" />
                <circle cx="27.5" cy="13.5" r="1" fill="white" />
              </svg>
              {/* Live ping dot */}
              <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_6px_rgba(34,211,238,0.8)]">
                <span className="absolute inset-0 rounded-full bg-cyan-400 animate-ping opacity-75" />
              </span>
            </div>
            <span className="text-[17px] font-bold tracking-[-0.04em] text-white">
              SatQuery<span className="text-cyan-400 ml-0.5">.AI</span>
            </span>
          </a>
          <span className="ml-1 px-2 py-0.5 rounded-full text-[10px] font-bold tracking-widest uppercase bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            PRO
          </span>
        </div>

        {/* Nav */}
        <nav className="hidden md:flex items-center gap-7 text-[13px] font-medium text-slate-400">
          <button onClick={() => navigate('/docs')} className="hover:text-white transition-colors cursor-pointer bg-transparent border-0 p-0">Analysis Docs</button>
          <button onClick={() => navigate('/benchmarks')} className="hover:text-white transition-colors cursor-pointer bg-transparent border-0 p-0">Benchmarks</button>
          <button onClick={() => navigate('/docs')} className="hover:text-white transition-colors cursor-pointer bg-transparent border-0 p-0">Docs</button>
        </nav>

        {/* Actions */}
        <div className="flex items-center gap-3">
          <button className="hidden sm:inline-flex items-center gap-2 px-3.5 py-1.5 text-xs font-medium text-slate-300 hover:text-white bg-white/[0.03] hover:bg-white/[0.08] border border-white/10 rounded-lg backdrop-blur-md transition-all cursor-pointer" type="button">
            <svg className="w-3.5 h-3.5 text-slate-400" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24">
              <circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" />
              <line x1="8.59" x2="15.42" y1="13.51" y2="17.49" /><line x1="15.41" x2="8.59" y1="6.51" y2="10.49" />
            </svg>
            Share
          </button>
          <button onClick={() => navigate('/workspace')} className="inline-flex items-center gap-2 px-4 py-2 rounded-lg font-semibold text-xs bg-gradient-to-r from-cyan-400 via-teal-400 to-emerald-400 text-slate-950 shadow-lg shadow-cyan-500/20 hover:brightness-105 active:scale-[0.98] transition-all duration-200 cursor-pointer" type="button">
            🚀 Launch Workspace
          </button>
        </div>
      </header>

      {/* ════ HERO ════ */}
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center text-center px-6 sm:px-10 lg:px-20 w-full pt-4 pb-8 pointer-events-auto">

        

        {/* Headline */}
        <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-[72px] font-extrabold tracking-[-0.04em] leading-[1.08] max-w-3xl text-white mx-auto">
          Ask anything about our<br />
          <span className="bg-gradient-to-r from-teal-200 via-cyan-300 to-indigo-300 bg-clip-text text-transparent">
            changing planet.
          </span>
        </h1>

        
        {/* Query Card */}
        <div className="mt-8 w-full max-w-2xl">
          <div
            className="rounded-2xl border border-white/10 bg-black/75 backdrop-blur-2xl p-4 relative overflow-hidden hover:border-white/20 transition-colors text-left flex flex-col"
            style={{ boxShadow: '0 0 50px -15px rgba(0,0,0,0.9), inset 0 1px 0 0 rgba(255,255,255,0.08)' }}
          >
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-400/30 to-transparent" />

            <div className="relative flex items-center justify-between min-h-[50px]">
              <div className="w-full">
                <label className="sr-only" htmlFor="query-input">Satellite AI Prompt</label>
                <textarea
                  id="query-input"
                  ref={queryRef}
                  rows={1}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleQueryOrbit();
                    }
                  }}
                  className="w-full resize-none border-0 bg-transparent p-0 text-sm sm:text-[15px] text-slate-100 placeholder-slate-600 focus:ring-0 focus:outline-none leading-relaxed font-normal tracking-[-0.01em] custom-scroll"
                  placeholder="Enter your query..."
                />
              </div>
            </div>

            <div className="mt-3 pt-3 border-t border-white/[0.06] flex flex-wrap items-center justify-between gap-2.5">
              <div className="flex flex-wrap items-center gap-2 relative">

                {/* Hidden file input */}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".tif,.tiff,.geotiff"
                  className="hidden"
                  onChange={handleFileUpload}
                />

                {/* Add Imagery button */}
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors cursor-pointer ${
                    uploadedFile
                      ? 'bg-emerald-500/10 border-emerald-400/30 text-emerald-300'
                      : 'bg-white/[0.03] hover:bg-white/[0.07] text-slate-300 border-white/10 hover:border-white/20'
                  }`}
                >
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" /><line x1="12" x2="12" y1="3" y2="15" />
                  </svg>
                  {uploadedFile ? uploadedFile.slice(0, 20) + (uploadedFile.length > 20 ? '…' : '') : 'Add Imagery (GeoTIFF)'}
                </button>

              </div>
              <button 
                type="button" 
                onClick={handleQueryOrbit} 
                disabled={isAnalyzing}
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-5 py-2 rounded-lg font-semibold text-xs sm:text-sm text-white bg-gradient-to-r from-purple-600 via-indigo-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 shadow-md active:scale-[0.98] transition-all duration-150 cursor-pointer disabled:opacity-50"
              >
                {isAnalyzing ? (
                  <>
                    <svg className="w-3.5 h-3.5 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.2" viewBox="0 0 24 24">
                      <circle cx="11" cy="11" r="8" /><line x1="21" x2="16.65" y1="21" y2="16.65" />
                    </svg>
                    Query Orbit
                  </>
                )}
              </button>
            </div>

            {/* AI Response Box */}
            {(isAnalyzing || aiResponse) && (
              <div className="mt-4 pt-4 border-t border-cyan-500/20 bg-cyan-950/10 rounded-xl p-4 shadow-inner text-left max-h-[300px] overflow-y-auto custom-scroll">
                <div className="flex items-center gap-2 mb-3">
                  <span className="material-symbols-outlined text-cyan-400 text-lg">psychology</span>
                  <span className="text-xs font-bold text-cyan-400 tracking-wider">SATQUERY AI</span>
                </div>
                {isAnalyzing ? (
                  <div className="space-y-2 opacity-60">
                    <div className="h-3 bg-slate-700/50 rounded w-3/4 animate-pulse"></div>
                    <div className="h-3 bg-slate-700/50 rounded w-full animate-pulse"></div>
                    <div className="h-3 bg-slate-700/50 rounded w-5/6 animate-pulse"></div>
                  </div>
                ) : (
                  <div className="text-sm font-sans text-slate-300 leading-relaxed prose prose-invert prose-cyan max-w-none">
                    <ReactMarkdown>{aiResponse || ''}</ReactMarkdown>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Fields of Interest */}
        <div className="mt-7 max-w-2xl mx-auto w-full">
          <p className="text-[11px] font-semibold text-swhite uppercase tracking-[0.15em] mb-3 text-center">Fields of Interest</p>
          <div className="flex flex-wrap justify-center gap-2">
            {[
              { emoji: '🌊', label: 'Coastal & Ocean Change',   color: 'cyan'    },
              { emoji: '🌿', label: 'Deforestation & LULC',     color: 'emerald' },
              { emoji: '🏙️', label: 'Urban Sprawl Detection',   color: 'violet'  },
              { emoji: '🔥', label: 'Wildfire & Burn Scars',    color: 'orange'  },
              { emoji: '❄️', label: 'Glacial Retreat',          color: 'sky'     },
              { emoji: '💧', label: 'Flood Mapping',            color: 'blue'    },
              { emoji: '⛏️', label: 'Mining & Extraction',      color: 'amber'   },
              { emoji: '🌾', label: 'Crop Health & Yield',      color: 'lime'    },
              { emoji: '🛢️', label: 'Oil Spill Detection',      color: 'rose'    },
              { emoji: '🚢', label: 'Maritime Surveillance',    color: 'indigo'  },
            ].map(({ emoji, label }) => (
              <button
                key={label}
                type="button"
                onClick={() => setQuery(label)}
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-medium border transition-all cursor-pointer backdrop-blur-md bg-black/30 hover:bg-white/[0.06] border-white/10 hover:border-cyan-400/40 text-slate-300 hover:text-white"
              >
                
                {label}
              </button>
            ))}
          </div>
        </div>
      </main>

      {/* ════ FOOTER BAR ════
      <footer className="relative z-20 w-full border-t border-white/[0.07] bg-black/80 backdrop-blur-2xl py-5 px-6 lg:px-12 pointer-events-auto">
        <div className="max-w-7xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-6 lg:gap-8">
          {[
            {
              color: 'teal', title: 'Multi-Sensor Data', sub: 'Sentinel-1/2, Landsat-9 & more',
              icon: <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24"><path d="M12 2a10 10 0 0 1 10 10" /><path d="M12 6a6 6 0 0 1 6 6" /><path d="M12 10a2 2 0 0 1 2 2" /><path d="M4 14l8-8" /><circle cx="4" cy="14" r="2" /></svg>,
            },
            {
              color: 'purple', title: 'AI-Powered Analysis', sub: 'Fast, accurate, reliable',
              icon: <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24"><rect height="16" rx="2" width="16" x="4" y="4" /><rect height="6" width="6" x="9" y="9" /><line x1="9" x2="9" y1="1" y2="4" /><line x1="15" x2="15" y1="1" y2="4" /><line x1="9" x2="9" y1="20" y2="23" /><line x1="15" x2="15" y1="20" y2="23" /><line x1="20" x2="23" y1="9" y2="9" /><line x1="20" x2="23" y1="15" y2="15" /><line x1="1" x2="4" y1="9" y2="9" /><line x1="1" x2="4" y1="15" y2="15" /></svg>,
            },
            {
              color: 'blue', title: 'Global Coverage', sub: 'Land, ocean & atmosphere',
              icon: <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" /><line x1="2" x2="22" y1="12" y2="12" /><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1 4-10z" /></svg>,
            },
            {
              color: 'emerald', title: 'Real-Time Insights', sub: 'For a sustainable tomorrow',
              icon: <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg>,
            },
          ].map(({ color, icon, title, sub }) => (
            <div key={title} className="flex items-center gap-3.5 group">
              <div className={`w-10 h-10 rounded-lg bg-${color}-500/10 border border-${color}-500/20 flex items-center justify-center text-${color}-400 group-hover:border-${color}-400/40 transition-colors`}>
                {icon}
              </div>
              <div>
                <h4 className="text-[13px] font-semibold text-white tracking-[-0.01em]">{title}</h4>
                <p className="text-xs text-slate-500 tracking-tight">{sub}</p>
              </div>
            </div>
          ))}
        </div>
      </footer> */}
    </div>
  );
}
