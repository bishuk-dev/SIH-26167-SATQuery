export type FailureOutcome =
  | "ALLOW"
  | "ALLOW_WITH_WARNING"
  | "REQUEST_INPUT"
  | "ABSTAIN"
  | "REJECT";

export type JobStatus =
  | "QUEUED"
  | "RUNNING"
  | "SUCCEEDED"
  | "FAILED"
  | "CANCEL_REQUESTED"
  | "CANCELLED"
  | "INTERRUPTED";

export type AnalysisStatus =
  | "PENDING"
  | "RUNNING"
  | "SUCCEEDED"
  | "FAILED"
  | "ABSTAINED"
  | "REJECTED"
  | "CANCELLED"
  | "INTERRUPTED";

interface ApiFailureEnvelope {
  error?: {
    code?: string;
    message?: string;
    outcome?: FailureOutcome;
    details?: Record<string, unknown>;
    request_id?: string;
  };
}

export class SatQueryApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly outcome?: FailureOutcome;
  readonly details: Record<string, unknown>;
  readonly requestId?: string;

  constructor(status: number, payload: ApiFailureEnvelope) {
    const failure = payload.error;
    super(failure?.message || `SatQuery API request failed with status ${status}.`);
    this.name = "SatQueryApiError";
    this.status = status;
    this.code = failure?.code || "API_REQUEST_FAILED";
    this.outcome = failure?.outcome;
    this.details = failure?.details || {};
    this.requestId = failure?.request_id;
  }
}

export interface ReadinessResponse {
  status: string;
  components: Record<string, { status?: string; [key: string]: unknown }>;
}

export interface ObservationResponse {
  observation_id: string;
  status: "READY";
  asset: {
    asset_id: string;
    original_name: string;
    sha256: string;
    immutable: true;
  };
  visualization: {
    asset_id: string;
    parent_asset_id: string;
    sha256: string;
    rendering: string;
    source_band_indexes: number[];
    tile_url_template: string;
    tile_scheme: string;
    tile_crs: string | null;
    tile_extent: { left: number; bottom: number; right: number; top: number };
    pixel_y_axis: "down" | null;
  };
  metadata: {
    raster: {
      driver: string;
      width: number;
      height: number;
      band_count: number;
      dtypes: string[];
      nodata: Array<number | null>;
      tags: Record<string, string>;
    };
    sensor: {
      modality: string;
      sensor_name: string | null;
      platform: string | null;
      product_level: string | null;
      bands: Array<{ index: number; description: string | null; dtype: string }>;
      polarizations: string[];
    };
    geo: {
      crs: string | null;
      bounds: { left: number; bottom: number; right: number; top: number } | null;
      native_gsd_x: number | null;
      native_gsd_y: number | null;
      units: string | null;
    };
    temporal: { acquisition_time: string | null };
    provenance: { created_at: string; ingestion_version: string };
  };
  validity: {
    has_crs: boolean;
    has_transform: boolean;
    has_nodata: boolean;
    metadata_quality: string;
    warnings: string[];
  };
  warnings: string[];
}

export interface QuerySubmissionResponse {
  analysis_id: string;
  job_id: string;
  status: JobStatus;
  plan_hash: string;
  registry_hash: string;
}

export interface JobResponse {
  job_id: string;
  analysis_id: string;
  status: JobStatus;
  created_at: string;
  updated_at: string;
}

export interface AnalysisResponse {
  analysis_id: string;
  status: AnalysisStatus;
  intent: string;
  created_at: string;
  updated_at: string;
  observation_ids: string[];
  plan_hash: string | null;
  registry_hash: string | null;
}

export interface TraceEvent {
  job_id: string;
  sequence: number;
  event_type: string;
  created_at: string;
  payload?: Record<string, unknown>;
}

export interface AnalysisReport {
  schema_version: 1;
  report_type: "analysis_report";
  analysis_id: string;
  status: AnalysisStatus;
  created_at: string;
  updated_at: string;
  query: string;
  intent: Record<string, unknown>;
  feasibility_outcome: FailureOutcome | null;
  inputs: {
    observation_ids: string[];
    pair_id: string | null;
    input_hashes: Record<string, string>;
  };
  workflow: {
    steps: Array<{
      step_id: string;
      tool_id: string;
      depends_on: string[];
      expected_evidence_type: string;
    }>;
    plan_hash: string | null;
    registry_hash: string | null;
  };
  answer: {
    answered: boolean;
    outcome: FailureOutcome;
    answer: string;
    evidence_ids: string[];
    measurements: Array<{
      evidence_id: string;
      measurement_type: string;
      value: number;
      unit: string;
      display_value: string;
      method: string;
      calculation_crs: string;
      positive_pixel_count: number;
      valid_pixel_count: number;
    }>;
    limitations: string[];
    uncalibrated_scores: Array<{ evidence_id: string; value: number; label: string }>;
  } | null;
  evidence: Array<{ evidence_id: string; task: string; artifact_ids: string[] }>;
  verification: {
    answered: boolean;
    passed: boolean;
    valid_evidence_ids: string[];
    invalid_evidence_ids: string[];
    issues: Array<{ code: string; severity: string; message: string; evidence_id: string | null }>;
    warnings: Array<{ code: string; severity: string; message: string; evidence_id: string | null }>;
  } | null;
  warnings: string[];
  trace: TraceEvent[];
  artifacts: Array<{
    artifact_id: string;
    evidence_id: string | null;
    media_type: string;
    sha256: string;
    size_bytes: number;
  }>;
  rerun_of: string | null;
}

const configuredBaseUrl = (import.meta.env.VITE_SATQUERY_API_BASE_URL || "").replace(/\/$/, "");

export function apiUrl(path: string) {
  return `${configuredBaseUrl}${path.startsWith("/") ? path : `/${path}`}`;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(apiUrl(path), {
      ...init,
      headers: {
        Accept: "application/json",
        ...init?.headers,
      },
    });
  } catch (error) {
    throw new Error(
      error instanceof Error
        ? `Cannot reach the SatQuery API: ${error.message}`
        : "Cannot reach the SatQuery API."
    );
  }

  const payload = (await response.json().catch(() => ({}))) as T & ApiFailureEnvelope;
  if (!response.ok) throw new SatQueryApiError(response.status, payload);
  return payload;
}

export async function getReadiness(signal?: AbortSignal): Promise<ReadinessResponse> {
  return requestJson<ReadinessResponse>("/api/v1/system/status", { signal });
}

export function uploadObservation(file: File): Promise<ObservationResponse> {
  const body = new FormData();
  body.append("file", file);
  return requestJson<ObservationResponse>("/api/v1/observations", { method: "POST", body });
}

export function submitQuery(
  query: string,
  observationIds: string[],
  idempotencyKey = crypto.randomUUID()
): Promise<QuerySubmissionResponse> {
  return requestJson<QuerySubmissionResponse>("/api/v1/query", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Idempotency-Key": idempotencyKey,
    },
    body: JSON.stringify({ query, observation_ids: observationIds }),
  });
}

export function rerunAnalysis(analysisId: string): Promise<QuerySubmissionResponse> {
  return requestJson<QuerySubmissionResponse>(`/api/v1/analyses/${analysisId}/rerun`, {
    method: "POST",
  });
}

export function getAnalysis(analysisId: string, signal?: AbortSignal): Promise<AnalysisResponse> {
  return requestJson<AnalysisResponse>(`/api/v1/analyses/${analysisId}`, { signal });
}

export function getJob(jobId: string, signal?: AbortSignal): Promise<JobResponse> {
  return requestJson<JobResponse>(`/api/v1/jobs/${jobId}`, { signal });
}

export function cancelJob(jobId: string): Promise<{ job: JobResponse }> {
  return requestJson<{ job: JobResponse }>(`/api/v1/jobs/${jobId}/cancel`, { method: "POST" });
}

export function getAnalysisReport(analysisId: string, signal?: AbortSignal): Promise<AnalysisReport> {
  return requestJson<AnalysisReport>(`/api/v1/reports/${analysisId}`, { signal });
}

export function getAnalysisTrace(
  analysisId: string,
  signal?: AbortSignal
): Promise<{ items: TraceEvent[] }> {
  return requestJson<{ items: TraceEvent[] }>(`/api/v1/analyses/${analysisId}/trace`, { signal });
}

export function getObservation(
  observationId: string,
  signal?: AbortSignal
): Promise<ObservationResponse> {
  return requestJson<ObservationResponse>(`/api/v1/observations/${observationId}`, { signal });
}
