"use client";

import { useState, useRef, useEffect, useCallback, lazy, Suspense } from "react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  ArrowUpIcon,
  Paperclip,
  Eye,
  Building2,
  TreePine,
  Layers,
  Activity,
  Waves,
  Radio,
  Satellite,
  X,
} from "lucide-react";

// Lazy-load Earth so Three.js doesn't block initial render
const EarthCanvas = lazy(() => import("@/components/ui/earth-canvas"));

/* ─── Auto-resize hook ─────────────────────────────────────────────────── */
interface AutoResizeProps {
  minHeight: number;
  maxHeight?: number;
}

function useAutoResizeTextarea({ minHeight, maxHeight }: AutoResizeProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const adjustHeight = useCallback(
    (reset?: boolean) => {
      const textarea = textareaRef.current;
      if (!textarea) return;
      if (reset) { textarea.style.height = `${minHeight}px`; return; }
      textarea.style.height = `${minHeight}px`;
      const newHeight = Math.max(minHeight, Math.min(textarea.scrollHeight, maxHeight ?? Infinity));
      textarea.style.height = `${newHeight}px`;
    },
    [minHeight, maxHeight]
  );

  useEffect(() => {
    if (textareaRef.current) textareaRef.current.style.height = `${minHeight}px`;
  }, [minHeight]);

  return { textareaRef, adjustHeight };
}

export interface AnalysisRequest {
  query: string;
  file: File;
}

interface RuixenMoonChatProps {
  onStartAnalysis: (request: AnalysisRequest) => Promise<void>;
  isSubmitting: boolean;
  submissionStage: string | null;
  submissionError: string | null;
  backendStatus: "checking" | "ready" | "degraded" | "offline";
}

function isTiff(file: File) {
  return /\.(tif|tiff|geotiff)$/i.test(file.name);
}

function formatFileSize(size: number) {
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

/* ─── Main component ───────────────────────────────────────────────────── */
export default function RuixenMoonChat({
  onStartAnalysis,
  isSubmitting,
  submissionStage,
  submissionError,
  backendStatus,
}: RuixenMoonChatProps) {
  const [message, setMessage] = useState("");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { textareaRef, adjustHeight } = useAutoResizeTextarea({
    minHeight: 48,
    maxHeight: 150,
  });

  const handleAppendQuery = (query: string) => {
    setMessage((prev) => (prev.trim() ? `${prev.trim()} ${query}` : query));
    textareaRef.current?.focus();
  };

  useEffect(() => {
    adjustHeight();
  }, [message, adjustHeight]);

  const selectFile = (file: File) => {
    if (!isTiff(file)) {
      setFileError("Choose a GeoTIFF file ending in .tif or .tiff.");
      return;
    }
    setUploadedFile(file);
    setFileError(null);
  };

  const handleSubmit = async () => {
    if (!message.trim() || !uploadedFile || isSubmitting) return;
    await onStartAnalysis({
      query: message.trim(),
      file: uploadedFile,
    });
  };

  return (
    <div className="relative w-full h-screen overflow-hidden flex flex-col items-center bg-[#050510]">
      {/* Hidden file input for .tif upload */}
      <input
        type="file"
        ref={fileInputRef}
        accept=".tif,.tiff,.geotiff,image/tiff"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) selectFile(file);
        }}
      />

      {/* Star field and vignette adapted from the apps/web2 landing background. */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage: "url('/earth/galaxy_starfield.png')",
          backgroundPosition: "center",
          backgroundSize: "cover",
          opacity: 0.58,
        }}
      />
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse at 50% 48%, transparent 22%, rgba(0,0,0,0.68) 82%)," +
            "linear-gradient(to bottom, rgba(2,3,15,0.18), rgba(5,5,25,0.72))",
        }}
      />

      {/* Top-right GitHub icon link */}
      <a
        href="https://github.com/bishuk-dev/SIH-26167-SATQuery"
        target="_blank"
        rel="noopener noreferrer"
        aria-label="GitHub Repository"
        className={cn(
          "absolute top-4 right-4 z-50",
          "p-2.5 rounded-full",
          "hover:scale-105",
          "transition-all duration-200 cursor-pointer"
        )}
      >
        <GithubIcon className="w-8 h-8" />
      </a>

      {/* ══════════════════════════════════════════════
          EARTH — centered, with room for orbital paths
      ══════════════════════════════════════════════ */}
      <div
        className="absolute pointer-events-none"
        style={{
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: "min(92vw, 92vh)",
          height: "min(92vw, 92vh)",
          zIndex: 1,
        }}
      >
        {/* Soft glow halo behind the Earth */}
        <div
          className="absolute inset-0 rounded-full"
          style={{
            background:
              "radial-gradient(circle, rgba(80,100,255,0.35) 0%, rgba(40,60,200,0.15) 50%, transparent 75%)",
            transform: "scale(1.25)",
            filter: "blur(30px)",
          }}
        />

        <Suspense fallback={null}>
          <EarthCanvas />
        </Suspense>
      </div>

      {/* ══════════════════════════════════════════════
          CONTENT LAYER — above the Earth canvas
      ══════════════════════════════════════════════ */}

      {/* Centered title — upper half */}
      <div
        className="relative flex-1 w-full flex flex-col items-center justify-center"
        style={{ zIndex: 10 }}
      >
        <div className="text-center select-none">
          <h1 className="text-5xl font-bold text-white tracking-tight drop-shadow-lg font-display">
            Sat Query
          </h1>
          <p className="mt-3 text-neutral-300 text-base font-light tracking-wide">
            Evidence-Backed Answers — From Intelligent Satellite Analysis.
          </p>
        </div>
      </div>

      {/* Input + quick actions — lower half, overlaid on Earth */}
      <div
        className="relative w-full max-w-3xl px-4 pb-10"
        style={{ zIndex: 10, marginBottom: "clamp(40px, 8vh, 100px)" }}
      >
        {/* Frosted-glass chat input */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDraggingFile(true);
          }}
          onDragLeave={() => setIsDraggingFile(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDraggingFile(false);
            const file = e.dataTransfer.files?.[0];
            if (file) selectFile(file);
          }}
          className={cn(
            "relative rounded-2xl border transition-all duration-200 shadow-2xl",
            isDraggingFile
              ? "border-sky-400/80 ring-2 ring-sky-400/30 bg-sky-950/40"
              : "border-white/10"
          )}
          style={{
            background: isDraggingFile ? undefined : "rgba(10, 10, 30, 0.55)",
            backdropFilter: "blur(20px)",
            WebkitBackdropFilter: "blur(20px)",
            boxShadow:
              "0 0 0 1px rgba(255,255,255,0.06), 0 20px 60px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08)",
          }}
        >
          {/* Uploaded File Chip Indicator */}
          {uploadedFile && (
            <div className="flex items-center gap-2 px-3 py-1.5 mx-3 mt-3 rounded-xl bg-sky-500/15 border border-sky-400/30 text-sky-300 text-xs shadow-sm animate-in fade-in duration-200">
              <div className="p-1 rounded bg-sky-400/20 text-sky-400">
                <Satellite className="w-3.5 h-3.5" />
              </div>
              <span className="font-mono font-medium truncate max-w-[280px]">
                {uploadedFile.name}
              </span>
              <span className="text-[11px] text-neutral-400 font-mono">
                ({formatFileSize(uploadedFile.size)})
              </span>
              <span className="px-1.5 py-0.5 rounded bg-sky-500/20 text-[9px] font-mono text-sky-300 font-bold tracking-wider uppercase">
                  Ready to upload
              </span>
              <button
                type="button"
                onClick={() => {
                  setUploadedFile(null);
                  if (fileInputRef.current) fileInputRef.current.value = "";
                }}
                className="ml-auto p-1 text-neutral-400 hover:text-white rounded hover:bg-white/10 transition-colors"
                title="Remove attached file"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          )}

          <Textarea
            ref={textareaRef}
            value={message}
            maxLength={500}
            onChange={(e) => {
              setMessage(e.target.value);
              adjustHeight();
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSubmit();
              }
            }}
            placeholder="Ask a question about your GeoTIFF..."
            className={cn(
              "w-full px-5 py-4 resize-none border-none bg-transparent",
              "text-white text-sm leading-relaxed font-sans",
              "focus-visible:ring-0 focus-visible:ring-offset-0",
              "placeholder:text-neutral-500 min-h-[52px]"
            )}
            style={{ overflow: "hidden" }}
          />

          {/* Toolbar row */}
          <div className="flex items-center justify-between px-4 pb-3 pt-1">
            <div className="flex items-center gap-2">
              {/* Tooltip wrapper */}
              <div className="relative group">
                <Button
                  variant="ghost"
                  size="icon"
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="text-neutral-400 hover:text-white hover:bg-white/10 rounded-lg w-8 h-8 cursor-pointer"
                >
                  <Paperclip className="w-4 h-4" />
                </Button>
                {/* Tooltip */}
                <div
                  className={cn(
                    "absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2.5 py-1.5",
                    "rounded-md text-xs font-medium whitespace-nowrap pointer-events-none",
                    "bg-neutral-800 text-neutral-200 border border-neutral-700",
                    "opacity-0 group-hover:opacity-100 scale-95 group-hover:scale-100",
                    "transition-all duration-150 ease-out",
                    "shadow-lg shadow-black/40 z-50"
                  )}
                >
                  Upload a .tif file
                  {/* Arrow */}
                  <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-neutral-800" />
                </div>
              </div>

              {/* A real file is required; the UI no longer fabricates a sample upload. */}
              {!uploadedFile && (
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="text-[11px] font-mono text-sky-400 hover:text-sky-300 px-2.5 py-1 rounded-lg bg-sky-500/10 hover:bg-sky-500/20 border border-sky-400/20 transition-all cursor-pointer flex items-center gap-1.5"
                >
                  <Satellite className="w-3 h-3 text-sky-400" />
                  <span>Select GeoTIFF</span>
                </button>
              )}
            </div>

            <Button
              type="button"
              onClick={handleSubmit}
              disabled={!message.trim() || !uploadedFile || isSubmitting}
              className={cn(
                "flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 cursor-pointer",
                message.trim() && uploadedFile && !isSubmitting
                  ? "bg-white text-black hover:bg-neutral-200 shadow-lg shadow-white/20 active:scale-95"
                  : "bg-white/10 text-neutral-500 cursor-not-allowed"
              )}
            >
              <ArrowUpIcon className="w-4 h-4" />
              <span className="sr-only">Send</span>
            </Button>
          </div>
        </div>

        <div className="mt-3 min-h-5 text-center text-xs" aria-live="polite">
          {(fileError || submissionError) && (
            <span className="text-rose-300">{fileError || submissionError}</span>
          )}
          {!fileError && !submissionError && submissionStage && (
            <span className="text-cyan-300">{submissionStage}</span>
          )}
          {!fileError && !submissionError && !submissionStage && (
            <span
              className={cn(
                "font-mono",
                backendStatus === "ready" && "text-emerald-400",
                backendStatus === "checking" && "text-neutral-400",
                backendStatus === "degraded" && "text-amber-300",
                backendStatus === "offline" && "text-rose-300"
              )}
            >
              API {backendStatus}
            </span>
          )}
        </div>

        {/* Quick query chips */}
        <div className="flex items-center justify-center flex-wrap gap-2.5 mt-3">
          <QuickAction
            icon={<Eye className="w-3.5 h-3.5" />}
            label="What do you see here?"
            onClick={() => handleAppendQuery("What do you see here?")}
          />
          <QuickAction
            icon={<Building2 className="w-3.5 h-3.5" />}
            label="Find buildings"
            onClick={() => handleAppendQuery("Find buildings")}
          />
          <QuickAction
            icon={<TreePine className="w-3.5 h-3.5" />}
            label="Measure vegetation"
            onClick={() => handleAppendQuery("Measure vegetation")}
          />
          <QuickAction
            icon={<Layers className="w-3.5 h-3.5" />}
            label="Compare these images"
            onClick={() => handleAppendQuery("Compare these images")}
          />
          <QuickAction
            icon={<Activity className="w-3.5 h-3.5" />}
            label="Detect changes"
            onClick={() => handleAppendQuery("Detect changes")}
          />
          <QuickAction
            icon={<Waves className="w-3.5 h-3.5" />}
            label="Analyze water extent"
            onClick={() => handleAppendQuery("Analyze water extent")}
          />
          <QuickAction
            icon={<Radio className="w-3.5 h-3.5" />}
            label="Compare Optical + SAR"
            onClick={() => handleAppendQuery("Compare Optical + SAR")}
          />
        </div>
      </div>
    </div>
  );
}

/* ─── QuickAction pill ──────────────────────────────────────────────────── */
interface QuickActionProps {
  icon: React.ReactNode;
  label: string;
  onClick?: () => void;
}

function QuickAction({ icon, label, onClick }: QuickActionProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex items-center gap-2 px-4 py-2 rounded-full text-xs font-medium",
        "border border-white/10 bg-white/5 text-neutral-300",
        "hover:bg-white/15 hover:text-white hover:border-white/20",
        "backdrop-blur-sm transition-all duration-200 cursor-pointer",
        "shadow-sm shadow-black/30"
      )}
    >
      {icon}
      {label}
    </button>
  );
}

/* ─── GitHub icon ──────────────────────────────────────────────────────── */
function GithubIcon({ className = "w-5 h-5" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="currentColor"
      className={className}
      aria-hidden="true"
    >
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
      />
    </svg>
  );
}
