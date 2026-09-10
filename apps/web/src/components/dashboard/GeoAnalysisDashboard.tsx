import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  Check,
  Copy,
  Download,
  ExternalLink,
  FileText,
  History,
  LoaderCircle,
  RefreshCw,
  Satellite,
  Square,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  apiUrl,
  cancelJob,
  getAnalysis,
  getAnalysisReport,
  getAnalysisTrace,
  getJob,
  getObservation,
  rerunAnalysis,
  SatQueryApiError,
  submitQuery,
  type AnalysisReport,
  type AnalysisResponse,
  type AnalysisStatus,
  type JobResponse,
  type ObservationResponse,
  type QuerySubmissionResponse,
  type TraceEvent,
} from "@/lib/satquery-api";

interface GeoAnalysisDashboardProps {
  analysisId: string;
  jobId?: string;
  uploadedFileName?: string;
  uploadedFileSize?: string;
  initialQuery?: string;
  initialObservation?: ObservationResponse;
  onBack: () => void;
  onAnalysisCreated: (submission: QuerySubmissionResponse) => void;
}

const TERMINAL_ANALYSIS_STATUSES = new Set<AnalysisStatus>([
  "SUCCEEDED",
  "FAILED",
  "ABSTAINED",
  "REJECTED",
  "CANCELLED",
  "INTERRUPTED",
]);

const glassPanelStyle = {
  background: "rgba(8, 12, 32, 0.78)",
  backdropFilter: "blur(20px)",
  WebkitBackdropFilter: "blur(20px)",
  boxShadow:
    "0 0 0 1px rgba(255,255,255,0.05), 0 20px 60px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.06)",
};

function formatDate(value?: string | null) {
  if (!value) return "Not reported";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function statusTone(status: string) {
  if (status === "SUCCEEDED") return "text-emerald-300 border-emerald-400/30 bg-emerald-400/10";
  if (["FAILED", "REJECTED", "INTERRUPTED"].includes(status)) {
    return "text-rose-300 border-rose-400/30 bg-rose-400/10";
  }
  if (["ABSTAINED", "CANCELLED", "CANCEL_REQUESTED"].includes(status)) {
    return "text-amber-300 border-amber-400/30 bg-amber-400/10";
  }
  return "text-sky-300 border-sky-400/30 bg-sky-400/10";
}

function errorMessage(error: unknown) {
  if (error instanceof SatQueryApiError) {
    return `${error.outcome ? `${error.outcome}: ` : ""}${error.message}`;
  }
  return error instanceof Error ? error.message : "The analysis request failed.";
}

function previewTileUrl(observation?: ObservationResponse) {
  if (!observation) return null;
  const { tile_url_template: template, tile_scheme: scheme, tile_extent: extent } =
    observation.visualization;
  let z = 0;
  let x = 0;
  let y = 0;

  if (scheme !== "pixel") {
    const halfWorld = 20_037_508.342789244;
    const span = Math.max(extent.right - extent.left, extent.top - extent.bottom);
    if (Number.isFinite(span) && span > 0) {
      z = Math.max(0, Math.min(14, Math.floor(Math.log2((halfWorld * 2) / span)) - 1));
      const scale = 2 ** z;
      const centerX = (extent.left + extent.right) / 2;
      const centerY = (extent.bottom + extent.top) / 2;
      x = Math.max(0, Math.min(scale - 1, Math.floor(((centerX + halfWorld) / (halfWorld * 2)) * scale)));
      y = Math.max(0, Math.min(scale - 1, Math.floor(((halfWorld - centerY) / (halfWorld * 2)) * scale)));
    }
  }

  return apiUrl(
    template.replace("{z}", String(z)).replace("{x}", String(x)).replace("{y}", String(y))
  );
}

export default function GeoAnalysisDashboard({
  analysisId,
  jobId,
  uploadedFileName = "Uploaded observation",
  uploadedFileSize,
  initialQuery = "",
  initialObservation,
  onBack,
  onAnalysisCreated,
}: GeoAnalysisDashboardProps) {
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [job, setJob] = useState<JobResponse | null>(null);
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [observation, setObservation] = useState<ObservationResponse | null>(
    initialObservation || null
  );
  const [trace, setTrace] = useState<TraceEvent[]>([]);
  const [query, setQuery] = useState(initialQuery);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [isActing, setIsActing] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [showTrace, setShowTrace] = useState(false);
  const [failedPreviewUrl, setFailedPreviewUrl] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let loadedObservationId = initialObservation?.observation_id;

    async function refresh() {
      try {
        const [nextAnalysis, nextReport, nextTrace] = await Promise.all([
          getAnalysis(analysisId, controller.signal),
          getAnalysisReport(analysisId, controller.signal),
          getAnalysisTrace(analysisId, controller.signal),
        ]);
        if (!active) return;

        const resolvedJobId = jobId || nextTrace.items.at(-1)?.job_id;
        const nextJob = resolvedJobId ? await getJob(resolvedJobId, controller.signal) : null;
        const observationId = nextReport.inputs.observation_ids[0] || nextAnalysis.observation_ids[0];
        let nextObservation: ObservationResponse | null = null;
        if (observationId && loadedObservationId !== observationId) {
          nextObservation = await getObservation(observationId, controller.signal);
          loadedObservationId = observationId;
        }
        if (!active) return;

        setAnalysis(nextAnalysis);
        setReport(nextReport);
        setTrace(nextTrace.items);
        if (nextJob) setJob(nextJob);
        if (nextObservation) setObservation(nextObservation);
        setQuery((current) => current || nextReport.query);
        setLoadError(null);

        if (!TERMINAL_ANALYSIS_STATUSES.has(nextAnalysis.status)) {
          timer = setTimeout(refresh, 1_200);
        }
      } catch (error) {
        if (!active || controller.signal.aborted) return;
        setLoadError(errorMessage(error));
        timer = setTimeout(refresh, 3_000);
      }
    }

    void refresh();
    return () => {
      active = false;
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [analysisId, initialObservation, jobId]);

  const status = job?.status || analysis?.status || "LOADING";
  const isProcessing = !analysis || !TERMINAL_ANALYSIS_STATUSES.has(analysis.status);
  const previewUrl = useMemo(() => previewTileUrl(observation || undefined), [observation]);
  const previewFailed = previewUrl !== null && failedPreviewUrl === previewUrl;
  const answer = report?.answer;
  const allWarnings = [
    ...(observation?.warnings || []),
    ...(report?.warnings || []),
    ...(answer?.limitations || []),
  ];

  async function handleRun() {
    if (!report || !query.trim() || isActing) return;
    setIsActing(true);
    setActionError(null);
    try {
      const submission =
        query.trim() === report.query.trim()
          ? await rerunAnalysis(analysisId)
          : await submitQuery(query.trim(), report.inputs.observation_ids);
      onAnalysisCreated(submission);
    } catch (error) {
      setActionError(errorMessage(error));
    } finally {
      setIsActing(false);
    }
  }

  async function handleCancel() {
    if (!job || isActing) return;
    setIsActing(true);
    setActionError(null);
    try {
      const response = await cancelJob(job.job_id);
      setJob(response.job);
    } catch (error) {
      setActionError(errorMessage(error));
    } finally {
      setIsActing(false);
    }
  }

  function downloadReport() {
    if (!report) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(report, null, 2)], { type: "application/json" })
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${report.analysis_id}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function copyLink() {
    await navigator.clipboard.writeText(window.location.href);
    setCopied(true);
    setTimeout(() => setCopied(false), 1_500);
  }

  return (
    <div className="relative flex h-screen w-screen flex-col overflow-hidden bg-[#050510] text-[#f0f6fc]">
      <div
        className="absolute inset-0 pointer-events-none opacity-35"
        style={{
          backgroundImage: "url('/earth/galaxy_starfield.png')",
          backgroundPosition: "center",
          backgroundSize: "cover",
        }}
      />
      <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(ellipse_at_center,rgba(24,52,110,0.2),rgba(5,5,16,0.92)_75%)]" />

      <header className="relative z-40 flex h-14 shrink-0 items-center justify-between border-b border-white/10 bg-[#080c20]/80 px-4 backdrop-blur-xl">
        <div className="flex min-w-0 items-center gap-3">
          <button type="button" onClick={onBack} className="rounded-lg p-2 text-neutral-400 hover:bg-white/10 hover:text-white" title="New analysis">
            <ArrowLeft className="h-4 w-4" />
          </button>
          <Satellite className="h-5 w-5 text-cyan-300" />
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold">SatQuery</span>
              <span className={cn("rounded-full border px-2 py-0.5 font-mono text-[10px]", statusTone(status))}>
                {status}
              </span>
            </div>
            <p className="truncate font-mono text-[10px] text-neutral-500">{analysisId}</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button type="button" onClick={copyLink} className="flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-neutral-300 hover:bg-white/10">
            {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
            <span className="hidden sm:inline">{copied ? "Copied" : "Share"}</span>
          </button>
          <button type="button" onClick={() => setShowReport(true)} disabled={!report} className="flex items-center gap-1.5 rounded-full bg-white px-3.5 py-1.5 text-xs font-medium text-black hover:bg-neutral-200 disabled:opacity-40">
            <FileText className="h-3.5 w-3.5" />
            Report
          </button>
        </div>
      </header>

      {(loadError || actionError) && (
        <div className="relative z-30 flex items-center gap-2 border-b border-rose-400/20 bg-rose-500/10 px-4 py-2 text-xs text-rose-200" role="alert">
          <AlertTriangle className="h-4 w-4 shrink-0" />
          <span>{actionError || loadError}</span>
        </div>
      )}

      <main className="relative z-10 flex min-h-0 flex-1">
        <aside style={glassPanelStyle} className="hidden w-80 shrink-0 flex-col border-r border-white/10 lg:flex">
          <div className="border-b border-white/10 p-4">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-neutral-300">Geo-query</span>
              <span className="text-[10px] text-neutral-500">500 characters max</span>
            </div>
            <textarea
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              maxLength={500}
              rows={4}
              className="w-full resize-none rounded-xl border border-white/10 bg-black/35 p-3 text-xs leading-relaxed text-white outline-none focus:border-cyan-400/40"
            />
            <div className="mt-2 flex gap-2">
              <button type="button" onClick={handleRun} disabled={!report || !query.trim() || isActing || isProcessing} className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-white/10 px-3 py-2 text-xs font-medium hover:bg-white/15 disabled:cursor-not-allowed disabled:opacity-40">
                <RefreshCw className={cn("h-3.5 w-3.5", isActing && "animate-spin")} />
                {query.trim() === report?.query.trim() ? "Rerun frozen plan" : "Run updated query"}
              </button>
              {isProcessing && job && (
                <button type="button" onClick={handleCancel} disabled={isActing} className="rounded-lg border border-white/10 p-2 text-neutral-400 hover:text-rose-300" title="Cancel job">
                  <Square className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
          </div>

          <div className="custom-scrollbar flex-1 space-y-3 overflow-y-auto p-4">
            <section className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
              <div className="mb-2 flex items-center gap-2 text-xs font-semibold text-white">
                {isProcessing ? <LoaderCircle className="h-4 w-4 animate-spin text-cyan-300" /> : <Activity className="h-4 w-4 text-emerald-300" />}
                Verified answer
              </div>
              {answer?.answered ? (
                <p className="text-xs leading-relaxed text-neutral-200">{answer.answer}</p>
              ) : (
                <p className="text-xs leading-relaxed text-neutral-400">
                  {isProcessing
                    ? "The backend is processing the frozen plan. No scientific answer is shown until verification completes."
                    : `No verified answer was produced${answer?.outcome ? ` (${answer.outcome})` : ""}.`}
                </p>
              )}
            </section>

            {answer?.measurements.map((measurement) => (
              <section key={measurement.evidence_id} className="rounded-2xl border border-cyan-400/20 bg-cyan-400/[0.05] p-4">
                <p className="text-[10px] uppercase tracking-wider text-cyan-300">{measurement.measurement_type}</p>
                <p className="mt-1 text-xl font-semibold text-white">{measurement.display_value}</p>
                <p className="mt-2 text-[10px] leading-relaxed text-neutral-400">{measurement.method} • {measurement.calculation_crs}</p>
              </section>
            ))}

            <section>
              <div className="mb-2 flex items-center justify-between text-[11px] uppercase tracking-wider text-neutral-400">
                <span>Evidence</span>
                <span>{report?.evidence.length || 0}</span>
              </div>
              <div className="space-y-2">
                {report?.evidence.map((item) => (
                  <div key={item.evidence_id} className="rounded-xl border border-white/10 bg-white/[0.025] p-3">
                    <p className="text-xs font-medium text-neutral-200">{item.task}</p>
                    <p className="mt-1 truncate font-mono text-[9px] text-neutral-500">{item.evidence_id}</p>
                  </div>
                ))}
                {!report?.evidence.length && <p className="text-xs text-neutral-500">No persisted evidence is available yet.</p>}
              </div>
            </section>

            {allWarnings.length > 0 && (
              <section className="rounded-2xl border border-amber-400/20 bg-amber-400/[0.05] p-3 text-[11px] leading-relaxed text-amber-100/80">
                <p className="mb-2 font-semibold text-amber-300">Warnings and limitations</p>
                <ul className="space-y-1.5">
                  {allWarnings.map((warning, index) => <li key={`${warning}-${index}`}>• {warning}</li>)}
                </ul>
              </section>
            )}
          </div>
        </aside>

        <section className="relative flex min-w-0 flex-1 flex-col bg-black/25">
          <div className="flex items-center justify-between border-b border-white/10 bg-black/20 px-4 py-2 text-[10px] text-neutral-400">
            <span>{observation ? `Display-only ${observation.visualization.rendering} visualization` : "Observation preview"}</span>
            <span className="font-mono">{observation?.visualization.tile_scheme || "waiting for metadata"}</span>
          </div>
          <div className="relative flex flex-1 items-center justify-center overflow-hidden p-5">
            <div className="absolute inset-0 opacity-20 [background-image:linear-gradient(rgba(56,189,248,.15)_1px,transparent_1px),linear-gradient(90deg,rgba(56,189,248,.15)_1px,transparent_1px)] [background-size:32px_32px]" />
            {previewUrl && !previewFailed ? (
              <img
                src={previewUrl}
                alt={`Display visualization for ${observation?.asset.original_name || "uploaded observation"}`}
                onError={() => setFailedPreviewUrl(previewUrl)}
                className="relative max-h-full max-w-full rounded-2xl border border-white/10 object-contain shadow-2xl shadow-black/60"
              />
            ) : (
              <div className="relative max-w-md rounded-3xl border border-white/10 bg-[#080c20]/75 p-8 text-center backdrop-blur-xl">
                {isProcessing ? <LoaderCircle className="mx-auto h-8 w-8 animate-spin text-cyan-300" /> : <Satellite className="mx-auto h-8 w-8 text-neutral-500" />}
                <p className="mt-4 text-sm font-medium text-neutral-200">
                  {previewFailed ? "The display tile could not be rendered." : "Preparing observation visualization…"}
                </p>
                <p className="mt-2 text-xs leading-relaxed text-neutral-500">
                  The original raster is never replaced by this display-only derivative.
                </p>
              </div>
            )}
          </div>
        </section>

        <aside style={glassPanelStyle} className="hidden w-80 shrink-0 flex-col border-l border-white/10 xl:flex">
          <div className="border-b border-white/10 p-4">
            <p className="text-xs font-semibold uppercase tracking-wider text-neutral-300">Observation metadata</p>
            <p className="mt-2 truncate text-xs text-white">{observation?.asset.original_name || uploadedFileName}</p>
            {uploadedFileSize && <p className="mt-1 text-[10px] text-neutral-500">Client upload: {uploadedFileSize}</p>}
          </div>
          <div className="custom-scrollbar flex-1 space-y-4 overflow-y-auto p-4 text-xs">
            <MetadataGroup title="Raster">
              <MetadataRow label="Driver" value={observation?.metadata.raster.driver} />
              <MetadataRow label="Dimensions" value={observation ? `${observation.metadata.raster.width} × ${observation.metadata.raster.height}` : undefined} />
              <MetadataRow label="Bands" value={observation?.metadata.raster.band_count} />
              <MetadataRow label="Data types" value={observation?.metadata.raster.dtypes.join(", ")} />
            </MetadataGroup>
            <MetadataGroup title="Sensor">
              <MetadataRow label="Modality" value={observation?.metadata.sensor.modality} />
              <MetadataRow label="Sensor" value={observation?.metadata.sensor.sensor_name} />
              <MetadataRow label="Platform" value={observation?.metadata.sensor.platform} />
              <MetadataRow label="Product" value={observation?.metadata.sensor.product_level} />
              <MetadataRow label="Polarizations" value={observation?.metadata.sensor.polarizations.join(", ")} />
            </MetadataGroup>
            <MetadataGroup title="Geospatial">
              <MetadataRow label="CRS" value={observation?.metadata.geo.crs} />
              <MetadataRow label="GSD X" value={observation?.metadata.geo.native_gsd_x} />
              <MetadataRow label="GSD Y" value={observation?.metadata.geo.native_gsd_y} />
              <MetadataRow label="Units" value={observation?.metadata.geo.units} />
              <MetadataRow label="Quality" value={observation?.validity.metadata_quality} />
            </MetadataGroup>
            <MetadataGroup title="Provenance">
              <MetadataRow label="Acquired" value={formatDate(observation?.metadata.temporal.acquisition_time)} />
              <MetadataRow label="Ingested" value={formatDate(observation?.metadata.provenance.created_at)} />
              <MetadataRow label="Ingestion version" value={observation?.metadata.provenance.ingestion_version} />
              <MetadataRow label="Source SHA-256" value={observation?.asset.sha256 ? `${observation.asset.sha256.slice(0, 16)}…` : undefined} />
            </MetadataGroup>
            <MetadataGroup title="Frozen workflow">
              <MetadataRow label="Intent" value={String(report?.intent.task_family || analysis?.intent || "Not reported")} />
              <MetadataRow label="Feasibility" value={report?.feasibility_outcome} />
              <MetadataRow label="Steps" value={report?.workflow.steps.length} />
              <MetadataRow label="Registry" value={report?.workflow.registry_hash ? `${report.workflow.registry_hash.slice(0, 16)}…` : undefined} />
            </MetadataGroup>
          </div>
        </aside>
      </main>

      <footer className="relative z-30 flex h-12 shrink-0 items-center justify-between border-t border-white/10 bg-[#080c20]/80 px-4 text-xs backdrop-blur-xl">
        <div className="flex min-w-0 items-center gap-3 text-neutral-400">
          <span className={cn("h-2 w-2 shrink-0 rounded-full", isProcessing ? "animate-pulse bg-cyan-400" : status === "SUCCEEDED" ? "bg-emerald-400" : "bg-amber-400")} />
          <span className="truncate">Updated {formatDate(job?.updated_at || analysis?.updated_at)}</span>
          {report?.verification && <span className="hidden sm:inline">Verification: {report.verification.passed ? "passed" : "not passed"}</span>}
        </div>
        <button type="button" onClick={() => setShowTrace(true)} className="flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-neutral-300 hover:bg-white/10">
          <History className="h-3.5 w-3.5" />
          Run log ({trace.length})
        </button>
      </footer>

      {showReport && report && (
        <Modal title="Analysis report" onClose={() => setShowReport(false)}>
          <div className="space-y-4 text-xs text-neutral-300">
            <div className="rounded-xl border border-white/10 bg-white/[0.035] p-4">
              <p className="font-mono text-[10px] text-neutral-500">{report.analysis_id}</p>
              <p className="mt-2 text-sm leading-relaxed text-white">{report.answer?.answer || "No verified answer was produced."}</p>
            </div>
            <div className="grid gap-2 sm:grid-cols-2">
              <ReportStat label="Outcome" value={report.answer?.outcome || report.status} />
              <ReportStat label="Verification" value={report.verification?.passed ? "PASSED" : "NOT PASSED"} />
              <ReportStat label="Evidence records" value={report.evidence.length} />
              <ReportStat label="Artifacts" value={report.artifacts.length} />
            </div>
            {report.artifacts.map((artifact) => (
              <div key={artifact.artifact_id} className="rounded-xl border border-white/10 p-3">
                <div className="flex justify-between gap-3"><span>{artifact.media_type}</span><span>{formatBytes(artifact.size_bytes)}</span></div>
                <p className="mt-1 truncate font-mono text-[9px] text-neutral-500">{artifact.artifact_id} • {artifact.sha256}</p>
              </div>
            ))}
            <div className="flex flex-wrap justify-end gap-2 border-t border-white/10 pt-4">
              <button type="button" onClick={() => window.open(apiUrl(`/api/v1/reports/${analysisId}/html`), "_blank", "noopener,noreferrer")} className="flex items-center gap-1.5 rounded-full border border-white/10 px-4 py-2 hover:bg-white/10">
                <ExternalLink className="h-3.5 w-3.5" /> HTML report
              </button>
              <button type="button" onClick={downloadReport} className="flex items-center gap-1.5 rounded-full bg-white px-4 py-2 font-medium text-black hover:bg-neutral-200">
                <Download className="h-3.5 w-3.5" /> Download JSON
              </button>
            </div>
          </div>
        </Modal>
      )}

      {showTrace && (
        <Modal title="Persisted execution trace" onClose={() => setShowTrace(false)}>
          <div className="space-y-2 font-mono text-[11px]">
            {trace.map((event) => (
              <div key={`${event.job_id}-${event.sequence}`} className="grid grid-cols-[3rem_1fr_auto] gap-3 rounded-xl border border-white/10 bg-white/[0.025] p-3">
                <span className="text-neutral-500">#{event.sequence}</span>
                <span className="text-cyan-200">{event.event_type}</span>
                <span className="text-neutral-500">{formatDate(event.created_at)}</span>
              </div>
            ))}
            {!trace.length && <p className="text-neutral-500">No execution events have been persisted yet.</p>}
          </div>
        </Modal>
      )}
    </div>
  );
}

function MetadataGroup({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-2xl border border-white/10 bg-white/[0.025] p-3">
      <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-neutral-400">{title}</p>
      <div className="space-y-2">{children}</div>
    </section>
  );
}

function MetadataRow({ label, value }: { label: string; value: unknown }) {
  const shown = value === null || value === undefined || value === "" ? "Not reported" : String(value);
  return <div className="flex justify-between gap-3"><span className="text-neutral-500">{label}</span><span className="min-w-0 truncate text-right font-mono text-[10px] text-neutral-200" title={shown}>{shown}</span></div>;
}

function ReportStat({ label, value }: { label: string; value: unknown }) {
  return <div className="rounded-xl border border-white/10 bg-white/[0.025] p-3"><p className="text-[10px] uppercase tracking-wider text-neutral-500">{label}</p><p className="mt-1 font-mono text-sm text-white">{String(value)}</p></div>;
}

function Modal({ title, onClose, children }: { title: string; onClose: () => void; children: React.ReactNode }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-md" role="dialog" aria-modal="true" aria-label={title}>
      <div style={glassPanelStyle} className="flex max-h-[85vh] w-full max-w-3xl flex-col overflow-hidden rounded-3xl border border-white/15">
        <div className="flex items-center justify-between border-b border-white/10 p-5">
          <h2 className="font-semibold text-white">{title}</h2>
          <button type="button" onClick={onClose} className="rounded-full p-1 text-neutral-400 hover:bg-white/10 hover:text-white" aria-label="Close">
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="custom-scrollbar overflow-y-auto p-5">{children}</div>
      </div>
    </div>
  );
}
