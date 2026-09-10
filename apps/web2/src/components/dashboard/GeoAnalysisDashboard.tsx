import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Satellite,
  Layers,
  Share2,
  Download,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Crosshair,
  Ruler,
  Info,
  History,
  BarChart3,
  FileText,
  Check,
  ChevronRight,
  Plus,
  RefreshCw,
  Sliders,
  Activity,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface GeoAnalysisDashboardProps {
  analysisId?: string;
  uploadedFileName?: string;
  uploadedFileSize?: string;
  initialQuery?: string;
  onBack: () => void;
}

export default function GeoAnalysisDashboard({
  analysisId,
  uploadedFileName = "barcelona_port_multisensor.tif",
  uploadedFileSize = "48.2 MB",
  initialQuery = "Assess coastal port infrastructure changes between T0 and T1, and identify unpermitted shoreline backfill.",
  onBack,
}: GeoAnalysisDashboardProps) {
  // ─── State ──────────────────────────────────────────────────────────────
  const [isLoading, setIsLoading] = useState(true);
  const [query, setQuery] = useState(initialQuery);
  const [isRerunning, setIsRerunning] = useState(false);
  const [sliderPosition, setSliderPosition] = useState(50); // percentage
  const [opticalOpacity, setOpticalOpacity] = useState(100);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [zoomLevel, setZoomLevel] = useState(1.0);
  const [activeTab, setActiveTab] = useState<"active" | "bands" | "metadata">("active");
  const [measureMode, setMeasureMode] = useState(false);
  const [activeReticle, setActiveReticle] = useState<number | null>(null);
  const [copiedShare, setCopiedShare] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1200);
    return () => clearTimeout(timer);
  }, []);

  const showSkeleton = isLoading || isRerunning;

  // Panels visibility
  const [leftPanelOpen, setLeftPanelOpen] = useState(true);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);

  // Feature overlay toggles
  const [showBoundaries, setShowBoundaries] = useState(true);
  const [showCorridors, setShowCorridors] = useState(true);
  const [showReclamation, setShowReclamation] = useState(true);

  // Modals
  const [showReportModal, setShowReportModal] = useState(false);
  const [showLogModal, setShowLogModal] = useState(false);
  const [showHistogramsModal, setShowHistogramsModal] = useState(false);
  const [showAddLayerModal, setShowAddLayerModal] = useState(false);

  // ─── Swipe Dragging Logic ────────────────────────────────────────────────
  const viewportRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDrag = useCallback((clientX: number) => {
    if (!viewportRef.current) return;
    const rect = viewportRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const clamped = Math.max(0, Math.min(x, rect.width));
    setSliderPosition((clamped / rect.width) * 100);
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      handleDrag(e.clientX);
    };
    const handleMouseUp = () => {
      if (isDragging) setIsDragging(false);
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging, handleDrag]);

  const handleTouchMove = (e: React.TouchEvent) => {
    if (!isDragging || !e.touches[0]) return;
    handleDrag(e.touches[0].clientX);
  };

  const handleRerun = () => {
    setIsRerunning(true);
    setTimeout(() => {
      setIsRerunning(false);
    }, 850);
  };

  const focusReticle = (id: number) => {
    setActiveReticle((prev) => (prev === id ? null : id));
  };

  const handleShare = () => {
    navigator.clipboard?.writeText(window.location.href);
    setCopiedShare(true);
    setTimeout(() => setCopiedShare(false), 2000);
  };

  // Glass style matching the home page frosted-glass container
  const glassPanelStyle = {
    background: "rgba(10, 10, 30, 0.72)",
    backdropFilter: "blur(20px)",
    WebkitBackdropFilter: "blur(20px)",
    boxShadow:
      "0 0 0 1px rgba(255,255,255,0.06), 0 20px 60px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08)",
  };

  return (
    <div className="relative w-screen h-screen overflow-hidden flex flex-col bg-[#050510] text-[#f0f6fc] font-sans select-none antialiased">
      {/* Background Radial Gradient matching Home Page */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 120% 80% at 50% 110%, #1a1060 0%, #050510 70%)",
        }}
      />

      {/* ══════════════════════════════════════════════════════════════════════
          1. TOP NAVIGATION BAR (Enlarged & Spacious)
      ══════════════════════════════════════════════════════════════════════ */}
      {/* ══════════════════════════════════════════════════════════════════════
          1. TOP NAVIGATION BAR (Height matches footer h-12)
      ══════════════════════════════════════════════════════════════════════ */}
      <header
        style={{
          background: "rgba(10, 10, 30, 0.7)",
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
        }}
        className="relative z-50 h-12 border-b border-white/10 px-4 flex items-center justify-between shrink-0 shadow-lg text-xs"
      >
        {/* Left branding & context */}
        <div className="flex items-center gap-3">
          {/* Logo & Brand with back action */}
          <div
            onClick={onBack}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onBack();
              }
            }}
            className="flex items-center gap-2 cursor-pointer select-none group"
            title="Return to Query Chat"
          >
            <Satellite className="w-4 h-4 text-white group-hover:scale-110 transition-transform" />
            <span className="text-sm font-semibold tracking-tight text-white group-hover:text-neutral-200 transition-colors">
              SatQuery
            </span>
          </div>

          <div className="h-4 w-px bg-white/15 hidden sm:block" />

          {/* Raster Context */}
          <div className="hidden md:flex items-center gap-2 text-xs text-neutral-300">
            {analysisId && (
              <>
                <span className="font-mono text-[10px] text-sky-300 bg-sky-500/10 border border-sky-400/20 px-2 py-0.5 rounded-full">
                  ID: {analysisId}
                </span>
                <span className="w-1 h-1 rounded-full bg-white/20" />
              </>
            )}
            <span className="text-neutral-500 font-light">Raster:</span>
            <span className="font-mono text-[11px] text-neutral-200 bg-white/5 border border-white/10 px-2.5 py-0.5 rounded-full">
              {uploadedFileName}
            </span>
            <span className="w-1 h-1 rounded-full bg-white/20" />
            <span className="text-neutral-400 font-mono text-[11px]">
              Sentinel-2 &amp; Sentinel-1 SAR
            </span>
          </div>
        </div>

        {/* Right actions */}
        <div className="flex items-center gap-2">
          {/* Status Pill */}
          <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded-full bg-white/5 border border-white/10 text-neutral-300 text-xs font-mono">
            {showSkeleton ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-ping shadow-[0_0_6px_#fbbf24]" />
                <span className="text-amber-300">Co-registering T0/T1...</span>
              </>
            ) : (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_6px_#34d399]" />
                <span>Ready • Co-registered T0/T1</span>
              </>
            )}
          </div>

          {/* Share Button */}
          <button
            type="button"
            onClick={handleShare}
            className="px-3 py-1 rounded-full border border-white/10 bg-white/5 hover:bg-white/10 text-neutral-300 hover:text-white text-xs font-medium flex items-center gap-1.5 transition-all cursor-pointer"
          >
            {copiedShare ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-400" />
                <span className="text-emerald-400">Copied!</span>
              </>
            ) : (
              <>
                <Share2 className="w-3.5 h-3.5" />
                <span className="hidden md:inline">Share</span>
              </>
            )}
          </button>

          {/* Export Button (White Pill CTA) */}
          <button
            type="button"
            onClick={() => setShowReportModal(true)}
            className="px-3.5 py-1 rounded-full bg-white text-black hover:bg-neutral-200 text-xs font-medium flex items-center gap-1.5 transition-all shadow-md shadow-white/10 active:scale-95 cursor-pointer"
          >
            <Download className="w-3.5 h-3.5 stroke-[2.2]" />
            <span>Export Analysis</span>
          </button>
        </div>
      </header>

      {/* ══════════════════════════════════════════════════════════════════════
          2. MAIN THREE-COLUMN WORKSPACE
      ══════════════════════════════════════════════════════════════════════ */}
      <main className="relative flex-1 flex overflow-hidden">
        {/* ─────────────────────────────────────────────────────────────
            LEFT PANEL: Geo-Query Assistant & Telemetry
        ───────────────────────────────────────────────────────────── */}
        <aside
          style={glassPanelStyle}
          className={cn(
            "relative z-30 flex flex-col border-r border-white/10 transition-all duration-300 shrink-0",
            leftPanelOpen ? "w-90 sm:w-96" : "w-0 overflow-hidden border-r-0"
          )}
        >
          {/* Query Header */}
          <div className="p-3.5 border-b border-white/10 bg-white/[0.02]">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-1.5 text-white">
                
                <span className="text-xs font-medium tracking-wide uppercase text-neutral-200">
                  Geo-Query Assistant
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="text-[10px] font-mono text-sky-300 px-2 py-0.5 rounded-full bg-sky-500/10 border border-sky-400/20">
                  AI ACTIVE
                </span>
                <button
                  type="button"
                  onClick={() => setLeftPanelOpen(false)}
                  className="p-1 text-neutral-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors cursor-pointer"
                  title="Collapse panel"
                >
                  <PanelLeftClose className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Prompt input */}
            <div className="relative">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                rows={3}
                className="w-full bg-black/40 border border-white/10 rounded-xl p-3 pb-9 text-xs text-neutral-200 placeholder:text-neutral-500 focus:outline-none focus:border-white/25 resize-none font-sans leading-relaxed shadow-inner"
                placeholder="Ask geospatial questions..."
              />
              <button
                type="button"
                onClick={handleRerun}
                disabled={isRerunning}
                title="Rerun Query"
                className="absolute bottom-2 right-2 px-3 py-1 rounded-lg bg-white/10 hover:bg-white/20 text-white font-medium text-[11px] border border-white/15 flex items-center gap-1.5 shadow-sm transition-all active:scale-95 disabled:opacity-50 cursor-pointer"
              >
                <RefreshCw
                  className={cn("w-3 h-3", isRerunning && "animate-spin")}
                />
                <span>{isRerunning ? "Analyzing..." : "Re-evaluate"}</span>
              </button>
            </div>
          </div>

          {/* Scrollable Telemetry Cards */}
          <div className="flex-1 overflow-y-auto p-3.5 space-y-3 custom-scrollbar">
            {showSkeleton ? (
              <div className="space-y-3 animate-pulse">
                {/* Detection Summary Skeleton */}
                <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-3.5 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-3.5 h-3.5 rounded-full bg-white/20" />
                      <div className="w-28 h-3.5 rounded bg-white/20" />
                    </div>
                    <div className="w-20 h-4 rounded-full bg-white/10" />
                  </div>
                  <div className="space-y-1.5 pt-1">
                    <div className="w-full h-3 rounded bg-white/15" />
                    <div className="w-4/5 h-3 rounded bg-white/10" />
                  </div>
                  <div className="w-full h-1.5 bg-white/10 rounded-full" />
                </div>

                {/* Evidence Points Skeleton */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between px-1">
                    <div className="w-32 h-3 rounded bg-white/10" />
                    <div className="w-14 h-3 rounded bg-white/10" />
                  </div>
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="p-3 rounded-xl border border-white/10 bg-white/[0.02] space-y-2.5">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className="w-4 h-4 rounded-full bg-white/15" />
                          <div className="w-36 h-3.5 rounded bg-white/20" />
                        </div>
                        <div className="w-12 h-3 rounded bg-white/10" />
                      </div>
                      <div className="flex justify-between pl-6">
                        <div className="w-24 h-2.5 rounded bg-white/10" />
                        <div className="w-20 h-2.5 rounded bg-white/10" />
                      </div>
                    </div>
                  ))}
                </div>

                {/* Context Note Skeleton */}
                <div className="p-3 rounded-xl bg-white/[0.02] border border-white/10 flex items-start gap-2.5">
                  <div className="w-4 h-4 rounded-full bg-white/15 shrink-0 mt-0.5" />
                  <div className="w-full space-y-1.5">
                    <div className="w-full h-2.5 rounded bg-white/10" />
                    <div className="w-3/4 h-2.5 rounded bg-white/10" />
                  </div>
                </div>
              </div>
            ) : (
              <>
                {/* Detection Summary Card */}
                <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-3.5 space-y-2.5 shadow-sm">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                      <Activity className="w-3.5 h-3.5 text-sky-400" />
                      <span className="text-xs font-medium text-white">
                        Detection Summary
                      </span>
                    </div>
                    <span className="text-[11px] font-mono text-neutral-200 bg-white/10 px-2 py-0.5 rounded-full border border-white/10">
                      94.2% Confidence
                    </span>
                  </div>
                  <p className="text-xs text-neutral-300 leading-relaxed font-light">
                    Built-up expansion detected in <strong className="text-white font-medium">Zone B-4</strong>. Radar backscatter increased by{" "}
                    <span className="text-sky-300 font-mono font-medium">+4.2 dB</span>, matching crane installations and eastern bulkhead reclamation.
                  </p>
                  <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-blue-500 to-sky-400 rounded-full w-[94.2%]"
                    />
                  </div>
                </div>

                {/* Evidence Points */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-[11px] font-medium tracking-wider uppercase text-neutral-400 px-1">
                    <span>Detected Evidence Points</span>
                    <span className="text-sky-300 font-mono lowercase">3 verified</span>
                  </div>

                  {/* Point 1 */}
                  <button
                    type="button"
                    onClick={() => focusReticle(1)}
                    className={cn(
                      "w-full text-left p-3 rounded-xl border transition-all group cursor-pointer",
                      activeReticle === 1
                        ? "border-white/30 bg-white/[0.08] shadow-lg shadow-black/40"
                        : "border-white/10 bg-white/[0.02] hover:bg-white/[0.05] hover:border-white/20"
                    )}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2">
                        <span className="w-4 h-4 rounded-full bg-white/10 border border-white/20 text-white font-mono text-[10px] font-medium flex items-center justify-center">
                          1
                        </span>
                        <span className="text-xs font-medium text-white">
                          Shoreline Bulkhead Backfill
                        </span>
                      </div>
                      <span className="font-mono text-[10px] text-neutral-400 group-hover:text-white flex items-center gap-0.5 group-hover:translate-x-0.5 transition-all">
                        Inspect <ChevronRight className="w-3 h-3" />
                      </span>
                    </div>
                    <div className="flex justify-between font-mono text-[11px] text-neutral-400 pl-6">
                      <span className="text-neutral-300">+34,800 m² reclaimed</span>
                      <span className="text-sky-300 font-medium">SAR Δ: +3.8 dB</span>
                    </div>
                  </button>

                  {/* Point 2 */}
                  <button
                    type="button"
                    onClick={() => focusReticle(2)}
                    className={cn(
                      "w-full text-left p-3 rounded-xl border transition-all group cursor-pointer",
                      activeReticle === 2
                        ? "border-white/30 bg-white/[0.08] shadow-lg shadow-black/40"
                        : "border-white/10 bg-white/[0.02] hover:bg-white/[0.05] hover:border-white/20"
                    )}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2">
                        <span className="w-4 h-4 rounded-full bg-white/10 border border-white/20 text-white font-mono text-[10px] font-medium flex items-center justify-center">
                          2
                        </span>
                        <span className="text-xs font-medium text-white">
                          Crane &amp; Rail Infrastructure
                        </span>
                      </div>
                      <span className="font-mono text-[10px] text-neutral-400 group-hover:text-white flex items-center gap-0.5 group-hover:translate-x-0.5 transition-all">
                        Inspect <ChevronRight className="w-3 h-3" />
                      </span>
                    </div>
                    <div className="flex justify-between font-mono text-[11px] text-neutral-400 pl-6">
                      <span className="text-neutral-300">New Gantry Cranes</span>
                      <span className="text-neutral-400">Berths 14-16</span>
                    </div>
                  </button>

                  {/* Point 3 */}
                  <button
                    type="button"
                    onClick={() => focusReticle(3)}
                    className={cn(
                      "w-full text-left p-3 rounded-xl border transition-all group cursor-pointer",
                      activeReticle === 3
                        ? "border-white/30 bg-white/[0.08] shadow-lg shadow-black/40"
                        : "border-white/10 bg-white/[0.02] hover:bg-white/[0.05] hover:border-white/20"
                    )}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2">
                        <span className="w-4 h-4 rounded-full bg-white/10 border border-white/20 text-white font-mono text-[10px] font-medium flex items-center justify-center">
                          3
                        </span>
                        <span className="text-xs font-medium text-white">
                          Shipping Fairway Vessel Traffic
                        </span>
                      </div>
                      <span className="font-mono text-[10px] text-neutral-400 group-hover:text-white flex items-center gap-0.5 group-hover:translate-x-0.5 transition-all">
                        Inspect <ChevronRight className="w-3 h-3" />
                      </span>
                    </div>
                    <div className="flex justify-between font-mono text-[11px] text-neutral-400 pl-6">
                      <span className="text-neutral-300">8 active cargo transits</span>
                      <span className="text-neutral-400">-14.2m channel</span>
                    </div>
                  </button>
                </div>

                {/* Tidal / Context Note */}
                <div className="p-3 rounded-xl bg-white/[0.03] border border-white/10 text-neutral-300 flex items-start gap-2 text-xs font-light">
                  <Info className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
                  <p className="text-[11px] leading-relaxed text-neutral-400">
                    Low-tide delta at T0 (-0.42m) compensated via EMODnet bathymetric baseline model. False positive shoreline shifts rejected.
                  </p>
                </div>
              </>
            )}
          </div>

          {/* Bottom Action (White pill matching home page) */}
          <div className="p-3.5 border-t border-white/10 bg-white/[0.02]">
            {showSkeleton ? (
              <div className="w-full h-10 rounded-xl bg-white/10 animate-pulse" />
            ) : (
              <button
                type="button"
                onClick={() => setShowReportModal(true)}
                className="w-full py-2.5 bg-white text-black hover:bg-neutral-200 font-medium text-xs rounded-xl flex items-center justify-center gap-2 transition-all shadow-lg shadow-white/10 active:scale-98 cursor-pointer"
              >
                <FileText className="w-3.5 h-3.5" />
                <span>Generate Change Report</span>
              </button>
            )}
          </div>
        </aside>

        {/* Collapsed Left Panel Opener */}
        {!leftPanelOpen && (
          <button
            type="button"
            onClick={() => setLeftPanelOpen(true)}
            className="absolute top-4 left-4 z-40 p-2.5 rounded-xl bg-[rgba(10,10,30,0.85)] border border-white/15 text-white hover:bg-white/10 backdrop-blur-xl shadow-xl transition-all cursor-pointer"
            title="Expand Assistant Panel"
          >
            <PanelLeftOpen className="w-4 h-4" />
          </button>
        )}

        {/* ─────────────────────────────────────────────────────────────
            CENTER HERO VIEWPORT: Interactive Dual-Sensor Swipe Canvas
        ───────────────────────────────────────────────────────────── */}
        <section className="relative flex-1 bg-[#050510] overflow-hidden flex items-center justify-center">
          <div
            ref={viewportRef}
            className="relative w-full h-full select-none overflow-hidden cursor-default"
            onTouchMove={handleTouchMove}
          >
            {/* Map Canvas Zoom & Pan Container */}
            <div
              className="w-full h-full relative transition-transform duration-200"
              style={{
                transform: `scale(${zoomLevel})`,
                transformOrigin: "center center",
              }}
            >
              {/* LAYER 1 (BOTTOM): Sentinel-1 SAR Radar Imagery */}
              <div className="absolute inset-0 w-full h-full">
                <img
                  src="/sentinel-1-sar.jpg"
                  alt="Sentinel-1 SAR C-band Radar Co-registered Imagery"
                  className="w-full h-full object-cover select-none pointer-events-none filter contrast-125 brightness-105"
                />

                {/* SAR Watermark Badge */}
                <div className="absolute top-4 right-4 bg-[rgba(10,10,30,0.85)] backdrop-blur-xl border border-white/15 px-3 py-1.5 rounded-full text-neutral-200 font-mono text-xs z-10 flex items-center gap-2 shadow-xl">
                  <span className="w-2 h-2 rounded-full bg-sky-400 animate-pulse shadow-[0_0_8px_#38bdf8]" />
                  <span>Sentinel-1 SAR (2024)</span>
                </div>
              </div>

              {/* LAYER 2 (TOP): Sentinel-2 Optical Imagery with Splitter Clip-Path */}
              <div
                className="absolute inset-0 w-full h-full overflow-hidden pointer-events-none"
                style={{
                  clipPath: `inset(0 ${100 - sliderPosition}% 0 0)`,
                  opacity: opticalOpacity / 100,
                  transition: isDragging ? "none" : "clip-path 0.1s ease-out",
                }}
              >
                <img
                  src="/sentinel-2-optical.jpg"
                  alt="Sentinel-2 True Color Optical Satellite Ingest"
                  className="absolute inset-0 w-full h-full object-cover select-none pointer-events-none"
                />

                {/* Optical Watermark Badge */}
                <div className="absolute top-4 left-4 bg-[rgba(10,10,30,0.85)] backdrop-blur-xl border border-white/15 px-3 py-1.5 rounded-full text-neutral-200 font-mono text-xs z-10 flex items-center gap-2 shadow-xl pointer-events-auto">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  <span>Sentinel-2 Optical (2023)</span>
                </div>
              </div>

              {/* Difference Heatmap / Feature Overlay Layer */}
              {showHeatmap && (
                <div
                  className="absolute inset-0 w-full h-full pointer-events-none z-15 mix-blend-screen transition-opacity duration-300"
                  style={{ opacity: showHeatmap ? 0.75 : 0 }}
                >
                  <svg className="w-full h-full" viewBox="0 0 1000 600" preserveAspectRatio="none">
                    {showReclamation && (
                      <path
                        d="M 380,270 Q 430,260 480,280 L 460,320 Q 400,310 380,270 Z"
                        fill="rgba(245, 158, 11, 0.4)"
                        stroke="#f59e0b"
                        strokeWidth="2"
                        strokeDasharray="4 2"
                        className="animate-pulse"
                      />
                    )}

                    {showBoundaries && (
                      <g stroke="#38bdf8" strokeWidth="2" fill="none" opacity="0.8">
                        <rect x="420" y="245" width="22" height="15" rx="2" fill="rgba(56, 189, 248, 0.25)" />
                        <rect x="450" y="245" width="22" height="15" rx="2" fill="rgba(56, 189, 248, 0.25)" />
                        <rect x="580" y="250" width="22" height="15" rx="2" fill="rgba(56, 189, 248, 0.25)" />
                        <rect x="610" y="250" width="22" height="15" rx="2" fill="rgba(56, 189, 248, 0.25)" />
                      </g>
                    )}

                    {showCorridors && (
                      <path
                        d="M 120,330 C 300,280 500,220 850,220"
                        stroke="rgba(6, 182, 212, 0.6)"
                        strokeWidth="2"
                        strokeDasharray="6 4"
                        fill="none"
                      />
                    )}
                  </svg>
                </div>
              )}

              {/* Co-registration / Calibration Scanner Overlay when showSkeleton */}
              {showSkeleton && (
                <div className="absolute inset-0 z-20 bg-black/40 backdrop-blur-[2px] flex flex-col items-center justify-center pointer-events-none transition-all duration-300">
                  <div className="relative w-44 h-44 rounded-full border border-sky-400/30 flex items-center justify-center">
                    <div className="absolute inset-0 rounded-full border border-sky-400/15 animate-ping" />
                    <div className="w-28 h-28 rounded-full border border-white/20 flex items-center justify-center">
                      <div className="w-14 h-14 rounded-full border border-sky-400/40 animate-pulse flex items-center justify-center bg-sky-500/10">
                        <Satellite className="w-6 h-6 text-sky-400 animate-spin" style={{ animationDuration: "3s" }} />
                      </div>
                    </div>
                    {/* Rotating radar sweep */}
                    <div className="absolute inset-0 origin-center animate-[spin_2.5s_linear_infinite] bg-gradient-to-tr from-sky-500/20 via-transparent to-transparent rounded-full" />
                  </div>

                  <div className="mt-5 px-4 py-2 rounded-full bg-[rgba(10,10,30,0.85)] border border-white/15 backdrop-blur-xl shadow-2xl flex items-center gap-2.5">
                    <span className="w-2 h-2 rounded-full bg-sky-400 animate-pulse shadow-[0_0_8px_#38bdf8]" />
                    <span className="text-xs font-mono text-neutral-200">
                      Co-registering S2 Optical &amp; S1 SAR rasters...
                    </span>
                  </div>
                  <div className="mt-2 text-[11px] font-mono text-neutral-400">
                    EPSG:32631 • Resampling 10m grid • Sigma-0 Calibration
                  </div>
                </div>
              )}

              {!showSkeleton && (
                <>
                  {/* ─── RETICLE MARKER 1: Bulkhead Reclamation ─── */}
                  <div
                    className={cn(
                      "absolute top-[44%] left-[36%] z-25 cursor-pointer transition-all duration-300 group",
                      activeReticle === 1 && "scale-115"
                    )}
                    onClick={() => focusReticle(1)}
                  >
                    <div className="relative">
                      <span className="w-6 h-6 rounded-full bg-[rgba(10,10,30,0.85)] backdrop-blur-md border border-white/40 text-white flex items-center justify-center text-xs font-medium shadow-xl hover:scale-110 transition-transform">
                        1
                      </span>
                      {/* Frosted Tooltip Card */}
                      <div
                        style={glassPanelStyle}
                        className={cn(
                          "absolute left-8 -top-3 w-56 p-3 rounded-2xl border border-white/15 transition-all z-30",
                          activeReticle === 1 ? "opacity-100 scale-100" : "opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto"
                        )}
                      >
                        <div className="flex items-center justify-between text-white text-xs font-medium mb-1">
                          <span>Bulkhead Reclamation</span>
                          <span className="text-[10px] font-mono bg-white/10 px-1.5 py-0.2 rounded-full text-neutral-300">
                            +34,800m²
                          </span>
                        </div>
                        <p className="text-[11px] text-neutral-300 font-light leading-relaxed">
                          Unpermitted shoreline infill identified between T0 &amp; T1. Radar backscatter shift +3.8 dB.
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* ─── RETICLE MARKER 2: Berth Expansion ─── */}
                  <div
                    className={cn(
                      "absolute top-[26%] left-[62%] z-25 cursor-pointer transition-all duration-300 group",
                      activeReticle === 2 && "scale-115"
                    )}
                    onClick={() => focusReticle(2)}
                  >
                    <div className="relative">
                      <span className="w-6 h-6 rounded-full bg-[rgba(10,10,30,0.85)] backdrop-blur-md border border-white/40 text-white flex items-center justify-center text-xs font-medium shadow-xl hover:scale-110 transition-transform">
                        2
                      </span>
                      {/* Frosted Tooltip Card */}
                      <div
                        style={glassPanelStyle}
                        className={cn(
                          "absolute left-8 -top-3 w-56 p-3 rounded-2xl border border-white/15 transition-all z-30",
                          activeReticle === 2 ? "opacity-100 scale-100" : "opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto"
                        )}
                      >
                        <div className="flex items-center justify-between text-white text-xs font-medium mb-1">
                          <span>Berth Expansion</span>
                          <span className="text-[10px] font-mono bg-white/10 px-1.5 py-0.2 rounded-full text-neutral-300">
                            +4.2 dB
                          </span>
                        </div>
                        <p className="text-[11px] text-neutral-300 font-light leading-relaxed">
                          SAR radar double-bounce: newly added rail-mounted gantry cranes.
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* ─── RETICLE MARKER 3: Fairway Channel ─── */}
                  <div
                    className={cn(
                      "absolute top-[68%] left-[64%] z-25 cursor-pointer transition-all duration-300 group",
                      activeReticle === 3 && "scale-115"
                    )}
                    onClick={() => focusReticle(3)}
                  >
                    <div className="relative">
                      <span className="w-6 h-6 rounded-full bg-[rgba(10,10,30,0.85)] backdrop-blur-md border border-white/40 text-white flex items-center justify-center text-xs font-medium shadow-xl hover:scale-110 transition-transform">
                        3
                      </span>
                      {/* Frosted Tooltip Card */}
                      <div
                        style={glassPanelStyle}
                        className={cn(
                          "absolute left-8 -top-3 w-56 p-3 rounded-2xl border border-white/15 transition-all z-30",
                          activeReticle === 3 ? "opacity-100 scale-100" : "opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto"
                        )}
                      >
                        <div className="flex items-center justify-between text-white text-xs font-medium mb-1">
                          <span>Fairway Channel</span>
                          <span className="text-[10px] font-mono bg-white/10 px-1.5 py-0.2 rounded-full text-neutral-300">
                            8 vessels
                          </span>
                        </div>
                        <p className="text-[11px] text-neutral-300 font-light leading-relaxed">
                          Verified acoustic and radar transit signatures at -14.2m channel depth.
                        </p>
                      </div>
                    </div>
                  </div>
                </>
              )}

              {/* Measure tool cursor indicator */}
              {measureMode && (
                <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                  <div className="px-3.5 py-2 rounded-full bg-black/80 backdrop-blur-md border border-white/20 text-neutral-200 font-mono text-xs flex items-center gap-2 shadow-2xl">
                    <Crosshair className="w-3.5 h-3.5 animate-spin" />
                    <span>Crosshair active: Click &amp; drag to measure distance</span>
                  </div>
                </div>
              )}
            </div>

            {/* ─── INTERACTIVE SWIPE SLIDER BAR ─── */}
            <div
              className="absolute top-0 bottom-0 w-px bg-white/50 shadow-[0_0_12px_rgba(255,255,255,0.4)] cursor-ew-resize z-30"
              style={{
                left: `${sliderPosition}%`,
                transition: isDragging ? "none" : "left 0.1s ease-out",
              }}
              onMouseDown={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onTouchStart={() => setIsDragging(true)}
            >
              {/* Drag Handle */}
              <div
                className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-[rgba(10,10,30,0.9)] backdrop-blur-xl border border-white/40 text-white flex items-center justify-center shadow-2xl pointer-events-auto hover:scale-110 active:scale-95 transition-transform cursor-ew-resize"
                title="Drag to compare Optical vs SAR"
              >
                <div className="flex items-center gap-1 text-[10px] font-bold text-neutral-300">
                  <span>‹</span>
                  <span>›</span>
                </div>
              </div>
            </div>

            {/* ─── BOTTOM HUD CONTROLS (Matching Home Page Pill Buttons) ─── */}
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-[rgba(10,10,30,0.75)] backdrop-blur-xl border border-white/10 px-2.5 py-1.5 rounded-full flex items-center gap-1.5 shadow-2xl z-30">
              <button
                type="button"
                onClick={() => setZoomLevel((prev) => Math.min(prev + 0.25, 2.5))}
                className="p-1.5 rounded-full hover:bg-white/10 text-neutral-300 hover:text-white transition-colors cursor-pointer"
                title="Zoom In"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              <button
                type="button"
                onClick={() => setZoomLevel((prev) => Math.max(prev - 0.25, 1.0))}
                className="p-1.5 rounded-full hover:bg-white/10 text-neutral-300 hover:text-white transition-colors cursor-pointer"
                title="Zoom Out"
              >
                <ZoomOut className="w-4 h-4" />
              </button>

              <div className="h-4 w-px bg-white/15" />

              <button
                type="button"
                onClick={() => setMeasureMode((prev) => !prev)}
                className={cn(
                  "p-1.5 rounded-full transition-colors cursor-pointer",
                  measureMode
                    ? "bg-white text-black"
                    : "hover:bg-white/10 text-neutral-300 hover:text-white"
                )}
                title="Measurement Tool"
              >
                <Ruler className="w-4 h-4" />
              </button>

              <button
                type="button"
                onClick={() => {
                  setZoomLevel(1.0);
                  setSliderPosition(50);
                }}
                className="p-1.5 rounded-full hover:bg-white/10 text-neutral-300 hover:text-white transition-colors cursor-pointer"
                title="Reset View Extent"
              >
                <Maximize2 className="w-4 h-4" />
              </button>

              <div className="h-4 w-px bg-white/15" />

              <button
                type="button"
                onClick={() => setShowHeatmap((prev) => !prev)}
                className={cn(
                  "px-3 py-1 rounded-full text-xs font-medium flex items-center gap-1.5 transition-all cursor-pointer",
                  showHeatmap
                    ? "bg-white text-black font-medium shadow-sm"
                    : "hover:bg-white/10 text-neutral-400 hover:text-white"
                )}
              >
                <span className={cn("w-2 h-2 rounded-full", showHeatmap ? "bg-amber-500" : "bg-neutral-500")} />
                <span>Heatmap</span>
              </button>
            </div>
          </div>
        </section>

        {/* Collapsed Right Panel Opener */}
        {!rightPanelOpen && (
          <button
            type="button"
            onClick={() => setRightPanelOpen(true)}
            className="absolute top-4 right-4 z-40 p-2.5 rounded-xl bg-[rgba(10,10,30,0.85)] border border-white/15 text-white hover:bg-white/10 backdrop-blur-xl shadow-xl transition-all cursor-pointer"
            title="Expand Layer Panel"
          >
            <PanelRightOpen className="w-4 h-4" />
          </button>
        )}

        {/* ─────────────────────────────────────────────────────────────
            RIGHT PANEL: Image Layers & Controls
        ───────────────────────────────────────────────────────────── */}
        <aside
          style={glassPanelStyle}
          className={cn(
            "relative z-30 flex flex-col border-l border-white/10 transition-all duration-300 shrink-0",
            rightPanelOpen ? "w-80 sm:w-84" : "w-0 overflow-hidden border-l-0"
          )}
        >
          {/* Header & Tabs */}
          <div className="p-3.5 border-b border-white/10 bg-white/[0.02]">
            <div className="flex items-center justify-between mb-2.5">
              <div className="flex items-center gap-2 text-white">
                <Layers className="w-4 h-4 text-sky-400" />
                <span className="text-xs font-medium tracking-wide uppercase text-neutral-200">
                  Image Layers
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="text-[11px] font-mono text-neutral-400">2 Co-registered</span>
                <button
                  type="button"
                  onClick={() => setRightPanelOpen(false)}
                  className="p-1 text-neutral-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors cursor-pointer"
                  title="Collapse panel"
                >
                  <PanelRightClose className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Sub-Tabs: Active, Bands, Metadata */}
            <div className="grid grid-cols-3 gap-1 bg-white/[0.04] p-1 rounded-xl border border-white/10 text-xs">
              <button
                type="button"
                onClick={() => setActiveTab("active")}
                className={cn(
                  "py-1.5 text-center font-medium rounded-lg transition-all cursor-pointer",
                  activeTab === "active"
                    ? "bg-white text-black font-medium shadow-sm"
                    : "text-neutral-400 hover:text-white"
                )}
              >
                Active
              </button>
              <button
                type="button"
                onClick={() => setActiveTab("bands")}
                className={cn(
                  "py-1.5 text-center font-medium rounded-lg transition-all cursor-pointer",
                  activeTab === "bands"
                    ? "bg-white text-black font-medium shadow-sm"
                    : "text-neutral-400 hover:text-white"
                )}
              >
                Bands
              </button>
              <button
                type="button"
                onClick={() => setActiveTab("metadata")}
                className={cn(
                  "py-1.5 text-center font-medium rounded-lg transition-all cursor-pointer",
                  activeTab === "metadata"
                    ? "bg-white text-black font-medium shadow-sm"
                    : "text-neutral-400 hover:text-white"
                )}
              >
                Metadata
              </button>
            </div>
          </div>

          {/* Scrollable Layers / Details */}
          <div className="flex-1 overflow-y-auto p-3.5 space-y-3 custom-scrollbar">
            {showSkeleton ? (
              <div className="space-y-3 animate-pulse">
                {/* Optical Baseline Skeleton */}
                <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-3.5 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-emerald-400/40" />
                      <div className="w-32 h-3.5 rounded bg-white/20" />
                    </div>
                    <div className="w-16 h-4 rounded-full bg-white/10" />
                  </div>
                  <div className="space-y-2 pt-1">
                    <div className="flex justify-between">
                      <div className="w-12 h-2.5 rounded bg-white/10" />
                      <div className="w-24 h-2.5 rounded bg-white/15" />
                    </div>
                    <div className="flex justify-between">
                      <div className="w-16 h-2.5 rounded bg-white/10" />
                      <div className="w-20 h-2.5 rounded bg-white/15" />
                    </div>
                    <div className="flex justify-between">
                      <div className="w-20 h-2.5 rounded bg-white/10" />
                      <div className="w-16 h-2.5 rounded bg-white/15" />
                    </div>
                  </div>
                </div>

                {/* SAR Observation Skeleton */}
                <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-3.5 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-sky-400/40" />
                      <div className="w-36 h-3.5 rounded bg-white/20" />
                    </div>
                    <div className="w-16 h-4 rounded-full bg-white/10" />
                  </div>
                  <div className="space-y-2 pt-1">
                    <div className="flex justify-between">
                      <div className="w-12 h-2.5 rounded bg-white/10" />
                      <div className="w-24 h-2.5 rounded bg-white/15" />
                    </div>
                    <div className="flex justify-between">
                      <div className="w-16 h-2.5 rounded bg-white/10" />
                      <div className="w-20 h-2.5 rounded bg-white/15" />
                    </div>
                    <div className="flex justify-between">
                      <div className="w-24 h-2.5 rounded bg-white/10" />
                      <div className="w-18 h-2.5 rounded bg-white/15" />
                    </div>
                  </div>
                </div>

                {/* Opacity Slider Skeleton */}
                <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-3.5 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="w-28 h-3 rounded bg-white/20" />
                    <div className="w-8 h-3 rounded bg-white/15" />
                  </div>
                  <div className="w-full h-1.5 rounded-lg bg-white/10" />
                  <div className="flex items-center justify-between pt-1">
                    <div className="w-28 h-3 rounded bg-white/10" />
                    <div className="w-9 h-5 rounded-full bg-white/10" />
                  </div>
                </div>

                {/* Feature Overlays Skeleton */}
                <div className="pt-1 space-y-2">
                  <div className="w-24 h-3 rounded bg-white/10 px-1" />
                  <div className="space-y-1.5">
                    {[1, 2, 3].map((i) => (
                      <div key={i} className="py-2.5 px-3 rounded-xl bg-white/[0.02] border border-white/5 flex items-center justify-between">
                        <div className="flex items-center gap-2.5">
                          <div className="w-3.5 h-3.5 rounded bg-white/15" />
                          <div className="w-32 h-3 rounded bg-white/15" />
                        </div>
                        <div className="w-14 h-2.5 rounded bg-white/10" />
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <>
                {activeTab === "active" && (
                  <>
                    {/* Optical Baseline Card */}
                    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-3.5 space-y-2 shadow-sm">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-emerald-400" />
                          <span className="text-xs font-medium text-white">
                            Optical Baseline (T0)
                          </span>
                        </div>
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-white/10 text-neutral-300">
                          Apr 2023
                        </span>
                      </div>
                      <div className="text-xs text-neutral-400 font-light space-y-1">
                        <div className="flex justify-between">
                          <span>Source</span>
                          <span className="text-neutral-200 font-mono text-[11px]">Sentinel-2B MSI</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Resolution</span>
                          <span className="text-neutral-200 font-mono text-[11px]">10m / px (RGB)</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Cloud Cover</span>
                          <span className="text-neutral-200 font-mono text-[11px]">0.4% (Clear)</span>
                        </div>
                      </div>
                    </div>

                    {/* SAR Observation Card */}
                    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-3.5 space-y-2 shadow-sm">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-sky-400" />
                          <span className="text-xs font-medium text-white">
                            Synthetic Aperture Radar (T1)
                          </span>
                        </div>
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-white/10 text-neutral-300">
                          Mar 2024
                        </span>
                      </div>
                      <div className="text-xs text-neutral-400 font-light space-y-1">
                        <div className="flex justify-between">
                          <span>Sensor</span>
                          <span className="text-neutral-200 font-mono text-[11px]">Sentinel-1 C-Band</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Polarization</span>
                          <span className="text-neutral-200 font-mono text-[11px]">VV + VH Dual</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Backscatter Shift</span>
                          <span className="text-sky-300 font-mono text-[11px] font-medium">
                            +4.2 dB Sigma-0
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Layer Opacity Slider */}
                    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-3.5 space-y-2.5 shadow-sm">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-medium text-white">Optical Layer Opacity</span>
                        <span className="text-neutral-300 font-mono text-[11px]">
                          {opticalOpacity}%
                        </span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={opticalOpacity}
                        onChange={(e) => setOpticalOpacity(Number(e.target.value))}
                        className="w-full h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer accent-white"
                      />

                      {/* Difference Heatmap Toggle */}
                      <div className="flex items-center justify-between pt-1 text-xs">
                        <div className="flex items-center gap-1.5 text-neutral-300 font-light">
                          <Sliders className="w-3.5 h-3.5 text-amber-400" />
                          <span>Highlight Change Areas</span>
                        </div>
                        <button
                          type="button"
                          onClick={() => setShowHeatmap((prev) => !prev)}
                          className={cn(
                            "w-9 h-5 rounded-full relative transition-colors duration-200 cursor-pointer shadow-inner",
                            showHeatmap ? "bg-white" : "bg-white/10"
                          )}
                        >
                          <span
                            className={cn(
                              "absolute top-0.5 w-4 h-4 rounded-full transition-all duration-200 shadow-sm",
                              showHeatmap ? "left-4.5 bg-black" : "left-0.5 bg-white"
                            )}
                          />
                        </button>
                      </div>
                    </div>

                    {/* Feature Overlays */}
                    <div className="pt-1 space-y-2">
                      <span className="text-[11px] font-medium tracking-wider uppercase text-neutral-400 px-1">
                        Feature Overlays
                      </span>
                      <div className="space-y-1.5 text-xs">
                        <label className="flex items-center justify-between py-2 px-3 rounded-xl bg-white/[0.02] hover:bg-white/[0.05] border border-white/5 cursor-pointer transition-all">
                          <div className="flex items-center gap-2.5">
                            <input
                              type="checkbox"
                              checked={showBoundaries}
                              onChange={(e) => setShowBoundaries(e.target.checked)}
                              className="rounded border-white/20 bg-black text-white focus:ring-0 w-3.5 h-3.5"
                            />
                            <span className="text-neutral-200 font-light">Terminals &amp; Port Boundary</span>
                          </div>
                          <span className="text-neutral-500 font-mono text-[11px]">14 zones</span>
                        </label>

                        <label className="flex items-center justify-between py-2 px-3 rounded-xl bg-white/[0.02] hover:bg-white/[0.05] border border-white/5 cursor-pointer transition-all">
                          <div className="flex items-center gap-2.5">
                            <input
                              type="checkbox"
                              checked={showCorridors}
                              onChange={(e) => setShowCorridors(e.target.checked)}
                              className="rounded border-white/20 bg-black text-white focus:ring-0 w-3.5 h-3.5"
                            />
                            <span className="text-neutral-200 font-light">Navigational Corridors</span>
                          </div>
                          <span className="text-neutral-500 font-mono text-[11px]">3 lanes</span>
                        </label>

                        <label className="flex items-center justify-between py-2 px-3 rounded-xl bg-white/[0.02] hover:bg-white/[0.05] border border-white/5 cursor-pointer transition-all">
                          <div className="flex items-center gap-2.5">
                            <input
                              type="checkbox"
                              checked={showReclamation}
                              onChange={(e) => setShowReclamation(e.target.checked)}
                              className="rounded border-white/20 bg-black text-white focus:ring-0 w-3.5 h-3.5"
                            />
                            <span className="text-neutral-200 font-light">Identified Reclamation</span>
                          </div>
                          <span className="text-amber-300 font-mono text-[11px] font-medium">+34.8k m²</span>
                        </label>
                      </div>
                    </div>
                  </>
                )}

                {activeTab === "bands" && (
                  <div className="space-y-3 text-xs">
                    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-3.5 space-y-2">
                      <span className="font-medium text-white">Sentinel-2 Spectral Bands</span>
                      <div className="space-y-1 text-neutral-300 font-mono text-[11px]">
                        <div className="flex justify-between py-1 border-b border-white/5">
                          <span>B02 (Blue 490 nm)</span>
                          <span className="text-neutral-300">10m</span>
                        </div>
                        <div className="flex justify-between py-1 border-b border-white/5">
                          <span>B03 (Green 560 nm)</span>
                          <span className="text-neutral-300">10m</span>
                        </div>
                        <div className="flex justify-between py-1 border-b border-white/5">
                          <span>B04 (Red 665 nm)</span>
                          <span className="text-neutral-300">10m</span>
                        </div>
                        <div className="flex justify-between py-1">
                          <span>B08 (NIR 842 nm)</span>
                          <span className="text-neutral-300">10m</span>
                        </div>
                      </div>
                    </div>

                    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-3.5 space-y-2">
                      <span className="font-medium text-white">Sentinel-1 Radar Polarization</span>
                      <div className="space-y-1 text-neutral-300 font-mono text-[11px]">
                        <div className="flex justify-between py-1 border-b border-white/5">
                          <span>VV (Vertical-Vertical)</span>
                          <span className="text-neutral-300">Calibrated</span>
                        </div>
                        <div className="flex justify-between py-1">
                          <span>VH (Vertical-Horizontal)</span>
                          <span className="text-neutral-300">Calibrated</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {activeTab === "metadata" && (
                  <div className="space-y-3 text-xs">
                    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-3.5 space-y-2">
                      <span className="font-medium text-white">Geospatial CRS &amp; Coordinates</span>
                      <div className="space-y-1.5 text-neutral-400 font-mono text-[11px]">
                        <div className="flex justify-between">
                          <span>CRS</span>
                          <span className="text-neutral-200">EPSG:32631</span>
                        </div>
                        <div className="flex justify-between">
                          <span>UTM Zone</span>
                          <span className="text-neutral-200">31N (WGS84)</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Bounding Box</span>
                          <span className="text-neutral-200 text-[10px]">2.14°E, 41.34°N</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Radiometric Unit</span>
                          <span className="text-sky-300">Sigma-0 (dB)</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Right panel bottom button */}
          <div className="p-3.5 border-t border-white/10 bg-white/[0.02]">
            {showSkeleton ? (
              <div className="w-full h-10 rounded-xl bg-white/10 animate-pulse" />
            ) : (
              <button
                type="button"
                onClick={() => setShowAddLayerModal(true)}
                className="w-full py-2.5 bg-white/5 hover:bg-white/10 text-neutral-200 hover:text-white border border-white/10 hover:border-white/20 font-medium text-xs rounded-xl flex items-center justify-center gap-1.5 transition-all cursor-pointer"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Add Imagery Layer</span>
              </button>
            )}
          </div>
        </aside>
      </main>

      {/* ══════════════════════════════════════════════════════════════════════
          3. BOTTOM TELEMETRY TRAY
      ══════════════════════════════════════════════════════════════════════ */}
      <footer
        style={{
          background: "rgba(10, 10, 30, 0.7)",
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
        }}
        className="h-12 border-t border-white/10 flex items-center justify-between px-4 z-40 shrink-0 text-xs"
      >
        <div className="flex items-center gap-3 text-neutral-400 font-light">
          {showSkeleton ? (
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping shadow-[0_0_6px_#fbbf24]" />
              <span className="text-amber-300 font-medium">Processing Pipeline...</span>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <span className="text-neutral-200 font-medium">Pipeline Complete</span>
            </div>
          )}

          <span className="text-white/20 hidden sm:inline">•</span>

          <div className="hidden sm:flex items-center gap-1 font-mono text-[11px]">
            <span>Center:</span>
            {showSkeleton ? (
              <span className="inline-block w-24 h-3.5 rounded bg-white/10 animate-pulse" />
            ) : (
              <span className="text-neutral-200">41°21'N 02°09'E</span>
            )}
          </div>

          <span className="text-white/20 hidden md:inline">•</span>

          <div className="hidden md:flex items-center gap-1 font-mono text-[11px]">
            <span>Projection:</span>
            {showSkeleton ? (
              <span className="inline-block w-28 h-3.5 rounded bg-white/10 animate-pulse" />
            ) : (
              <span className="text-neutral-200">UTM 31N (WGS84)</span>
            )}
          </div>

          <span className="text-white/20 hidden lg:inline">•</span>

          <div className="hidden lg:flex items-center gap-1 font-mono text-[11px]">
            <span>Latency:</span>
            {showSkeleton ? (
              <span className="inline-block w-14 h-3.5 rounded bg-white/10 animate-pulse" />
            ) : (
              <span className="text-emerald-400 font-semibold">842ms</span>
            )}
          </div>
        </div>

        {/* Modal Triggers */}
        <div className="flex items-center gap-2">
          {showSkeleton ? (
            <>
              <div className="w-20 h-7 rounded-full bg-white/10 animate-pulse" />
              <div className="w-24 h-7 rounded-full bg-white/10 animate-pulse" />
            </>
          ) : (
            <>
              <button
                type="button"
                onClick={() => setShowLogModal(true)}
                className="px-3 py-1 rounded-full bg-white/5 border border-white/10 hover:bg-white/10 text-neutral-300 hover:text-white text-xs transition-colors flex items-center gap-1.5 cursor-pointer"
              >
                <History className="w-3.5 h-3.5" />
                <span>Run Log</span>
              </button>

              <button
                type="button"
                onClick={() => setShowHistogramsModal(true)}
                className="px-3 py-1 rounded-full bg-white/5 border border-white/10 hover:bg-white/10 text-neutral-300 hover:text-white text-xs transition-colors flex items-center gap-1.5 cursor-pointer"
              >
                <BarChart3 className="w-3.5 h-3.5" />
                <span>Histograms</span>
              </button>
            </>
          )}
        </div>
      </footer>

      {/* ══════════════════════════════════════════════════════════════════════
          4. MODALS (Styled with matching frosted glass)
      ══════════════════════════════════════════════════════════════════════ */}

      {/* Change Report Modal */}
      {showReportModal && (
        <div className="fixed inset-0 z-100 bg-black/75 backdrop-blur-md flex items-center justify-center p-4">
          <div
            style={glassPanelStyle}
            className="relative w-full max-w-2xl border border-white/15 rounded-3xl p-6 overflow-hidden flex flex-col max-h-[85vh] shadow-2xl"
          >
            <div className="flex items-center justify-between pb-4 border-b border-white/10">
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-sky-400" />
                <h3 className="text-base font-semibold text-white">
                  SatQuery Change Detection Verification Report {analysisId ? `• ${analysisId}` : ""}
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setShowReportModal(false)}
                className="p-1 rounded-full text-neutral-400 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto py-4 space-y-4 text-xs leading-relaxed text-neutral-300 font-light custom-scrollbar">
              <div className="p-3.5 rounded-2xl bg-white/[0.03] border border-white/10 text-neutral-200">
                <strong className="text-white font-medium">Executive Assessment:</strong> Significant port expansion confirmed across Zone B-4. High confidence multi-sensor consensus between Optical NIR loss and SAR Sigma-0 backscatter surge.
              </div>

              <div className="space-y-2">
                <h4 className="font-medium text-white text-sm">Quantitative Derived Metrics</h4>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 font-mono text-[11px]">
                  <div className="p-3 rounded-xl bg-white/5 border border-white/10">
                    <div className="text-neutral-400 text-[10px] font-sans">Net Reclaimed Land</div>
                    <div className="text-white text-sm font-semibold mt-1">+34,800 m²</div>
                  </div>
                  <div className="p-3 rounded-xl bg-white/5 border border-white/10">
                    <div className="text-neutral-400 text-[10px] font-sans">Radar Backscatter Shift</div>
                    <div className="text-sky-300 text-sm font-semibold mt-1">+4.2 dB Sigma-0</div>
                  </div>
                  <div className="p-3 rounded-xl bg-white/5 border border-white/10">
                    <div className="text-neutral-400 text-[10px] font-sans">Statistical Certainty</div>
                    <div className="text-emerald-400 text-sm font-semibold mt-1">p &lt; 0.001</div>
                  </div>
                </div>
              </div>

              <div className="space-y-1">
                <h4 className="font-medium text-white text-sm">Sensor Provenance &amp; Calibration</h4>
                <p className="text-neutral-400">
                  - T0 Optical: Sentinel-2B MSI (Level-2A BOA), EPSG:32631, 10m GSD, clear cloud condition.<br />
                  - T1 Radar: Sentinel-1 IW C-band GRD (Terrain Corrected Range Doppler), VV/VH dual polarization.
                </p>
              </div>
            </div>

            <div className="pt-4 border-t border-white/10 flex justify-end gap-2.5">
              <button
                type="button"
                onClick={() => setShowReportModal(false)}
                className="px-4 py-2 rounded-full border border-white/10 bg-white/5 hover:bg-white/10 text-neutral-300 text-xs font-medium cursor-pointer"
              >
                Close
              </button>
              <button
                type="button"
                onClick={() => {
                  alert("Report JSON and GeoTIFF mask exported to download folder.");
                  setShowReportModal(false);
                }}
                className="px-4 py-2 rounded-full bg-white text-black hover:bg-neutral-200 font-medium text-xs flex items-center gap-1.5 shadow-lg shadow-white/10 cursor-pointer"
              >
                <Download className="w-3.5 h-3.5 stroke-[2.2]" />
                <span>Download PDF &amp; GeoJSON</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Run Log Modal */}
      {showLogModal && (
        <div className="fixed inset-0 z-100 bg-black/75 backdrop-blur-md flex items-center justify-center p-4">
          <div
            style={glassPanelStyle}
            className="relative w-full max-w-xl border border-white/15 rounded-3xl p-6 flex flex-col max-h-[80vh] shadow-2xl"
          >
            <div className="flex items-center justify-between pb-3 border-b border-white/10">
              <div className="flex items-center gap-2">
                <History className="w-4 h-4 text-sky-400" />
                <h3 className="text-sm font-semibold text-white">Pipeline Execution Telemetry</h3>
              </div>
              <button
                type="button"
                onClick={() => setShowLogModal(false)}
                className="p-1 rounded-full text-neutral-400 hover:text-white cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto py-3 space-y-2 font-mono text-[11px] text-neutral-300 custom-scrollbar">
              <div className="text-neutral-500">[00:00:01] Ingesting GeoTIFF payload: {uploadedFileName}</div>
              <div className="text-neutral-500">[00:00:12] GDAL inspect: 4 spectral bands detected, CRS EPSG:32631</div>
              <div className="text-neutral-500">[00:00:24] Sentinel-2 Level-2A TOA reflectance normalizer initialized</div>
              <div className="text-neutral-500">[00:00:45] Sentinel-1 SAR C-band Sigma-0 radiometry aligned</div>
              <div className="text-neutral-300">[00:01:10] Co-registration affine warp: Residual RMSE = 0.18 pixels</div>
              <div className="text-sky-300">[00:01:52] Change detector: +4.2 dB backscatter delta identified</div>
              <div className="text-emerald-400">[00:02:18] Execution clean: Deterministic land area derived: 34,800 m²</div>
            </div>

            <div className="pt-3 border-t border-white/10 flex justify-end">
              <button
                type="button"
                onClick={() => setShowLogModal(false)}
                className="px-4 py-1.5 rounded-full bg-white/10 hover:bg-white/15 text-white text-xs cursor-pointer"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Histograms Modal */}
      {showHistogramsModal && (
        <div className="fixed inset-0 z-100 bg-black/75 backdrop-blur-md flex items-center justify-center p-4">
          <div
            style={glassPanelStyle}
            className="relative w-full max-w-xl border border-white/15 rounded-3xl p-6 flex flex-col shadow-2xl"
          >
            <div className="flex items-center justify-between pb-3 border-b border-white/10">
              <div className="flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-sky-400" />
                <h3 className="text-sm font-semibold text-white">Radiometric Histogram Distributions</h3>
              </div>
              <button
                type="button"
                onClick={() => setShowHistogramsModal(false)}
                className="p-1 rounded-full text-neutral-400 hover:text-white cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="py-4 space-y-4">
              <div>
                <div className="flex justify-between text-xs mb-1 font-mono text-neutral-400">
                  <span>SAR Backscatter Sigma-0 (dB)</span>
                  <span className="text-sky-300">Shift +4.2 dB</span>
                </div>
                <div className="h-24 bg-white/[0.03] border border-white/10 rounded-2xl p-2 flex items-end">
                  <svg className="w-full h-full" viewBox="0 0 300 80">
                    <path
                      d="M 10,75 Q 80,70 120,40 T 180,10 T 230,45 T 290,75"
                      fill="rgba(56, 189, 248, 0.15)"
                      stroke="#38bdf8"
                      strokeWidth="2"
                    />
                  </svg>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1 font-mono text-neutral-400">
                  <span>Optical NDVI Water vs Land Distribution</span>
                  <span className="text-emerald-400">Bimodal Split: 0.12</span>
                </div>
                <div className="h-24 bg-white/[0.03] border border-white/10 rounded-2xl p-2 flex items-end">
                  <svg className="w-full h-full" viewBox="0 0 300 80">
                    <path
                      d="M 10,75 Q 60,15 100,70 T 190,20 T 290,75"
                      fill="rgba(52, 211, 153, 0.15)"
                      stroke="#34d399"
                      strokeWidth="2"
                    />
                  </svg>
                </div>
              </div>
            </div>

            <div className="pt-3 border-t border-white/10 flex justify-end">
              <button
                type="button"
                onClick={() => setShowHistogramsModal(false)}
                className="px-4 py-1.5 rounded-full bg-white/10 hover:bg-white/15 text-white text-xs cursor-pointer"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Add Layer Modal */}
      {showAddLayerModal && (
        <div className="fixed inset-0 z-100 bg-black/75 backdrop-blur-md flex items-center justify-center p-4">
          <div
            style={glassPanelStyle}
            className="relative w-full max-w-md border border-white/15 rounded-3xl p-6 flex flex-col shadow-2xl"
          >
            <div className="flex items-center justify-between pb-3 border-b border-white/10">
              <div className="flex items-center gap-2">
                <Plus className="w-4 h-4 text-sky-400" />
                <h3 className="text-sm font-semibold text-white">Add Co-registered Layer</h3>
              </div>
              <button
                type="button"
                onClick={() => setShowAddLayerModal(false)}
                className="p-1 rounded-full text-neutral-400 hover:text-white cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="py-4 space-y-3 text-xs">
              <label className="block p-3 rounded-2xl border border-white/10 bg-white/5 hover:bg-white/10 cursor-pointer transition-all">
                <input type="radio" name="layerChoice" defaultChecked className="mr-2.5 text-white" />
                <span className="text-white font-medium">Landsat-9 OLI-2 (30m Optical Multispectral)</span>
              </label>
              <label className="block p-3 rounded-2xl border border-white/10 bg-white/5 hover:bg-white/10 cursor-pointer transition-all">
                <input type="radio" name="layerChoice" className="mr-2.5 text-white" />
                <span className="text-white font-medium">PlanetScope High-Res (3m Orthotile)</span>
              </label>
              <label className="block p-3 rounded-2xl border border-white/10 bg-white/5 hover:bg-white/10 cursor-pointer transition-all">
                <input type="radio" name="layerChoice" className="mr-2.5 text-white" />
                <span className="text-white font-medium">EMODnet Bathymetry Depth Contours</span>
              </label>
            </div>

            <div className="pt-3 border-t border-white/10 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setShowAddLayerModal(false)}
                className="px-4 py-1.5 rounded-full border border-white/10 bg-white/5 text-neutral-300 text-xs cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => {
                  alert("Raster layer loaded into workspace.");
                  setShowAddLayerModal(false);
                }}
                className="px-4 py-1.5 rounded-full bg-white text-black hover:bg-neutral-200 font-medium text-xs cursor-pointer"
              >
                Load Layer
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
