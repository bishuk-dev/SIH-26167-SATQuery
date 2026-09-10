import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import RuixenMoonChat, { type AnalysisRequest } from "@/components/ui/ruixen-moon-chat";
import {
  getReadiness,
  SatQueryApiError,
  submitQuery,
  uploadObservation,
} from "@/lib/satquery-api";

export default function HomePage() {
  const navigate = useNavigate();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submissionStage, setSubmissionStage] = useState<string | null>(null);
  const [submissionError, setSubmissionError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<
    "checking" | "ready" | "degraded" | "offline"
  >("checking");

  useEffect(() => {
    const controller = new AbortController();
    getReadiness(controller.signal)
      .then((readiness) => setBackendStatus(readiness.status === "ready" ? "ready" : "degraded"))
      .catch(() => {
        if (!controller.signal.aborted) setBackendStatus("offline");
      });
    return () => controller.abort();
  }, []);

  const handleStartAnalysis = async (request: AnalysisRequest) => {
    setIsSubmitting(true);
    setSubmissionError(null);

    try {
      setSubmissionStage("Uploading and validating GeoTIFF…");
      const observation = await uploadObservation(request.file);

      setSubmissionStage("Planning and queueing evidence-grounded analysis…");
      const submission = await submitQuery(request.query, [observation.observation_id]);

      navigate(`/analysis/${submission.analysis_id}?job=${submission.job_id}`, {
        state: {
          query: request.query,
          file: {
            name: request.file.name,
            size: request.file.size,
          },
          observation,
        },
      });
    } catch (error) {
      const prefix = error instanceof SatQueryApiError && error.outcome ? `${error.outcome}: ` : "";
      setSubmissionError(`${prefix}${error instanceof Error ? error.message : "Analysis submission failed."}`);
      setSubmissionStage(null);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="h-screen w-full bg-[#050510] text-white overflow-hidden select-none">
      <RuixenMoonChat
        onStartAnalysis={handleStartAnalysis}
        isSubmitting={isSubmitting}
        submissionStage={submissionStage}
        submissionError={submissionError}
        backendStatus={backendStatus}
      />
    </main>
  );
}
