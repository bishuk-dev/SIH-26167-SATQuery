import React from "react";
import { Navigate, useParams, useLocation, useNavigate, useSearchParams } from "react-router-dom";
import GeoAnalysisDashboard from "@/components/dashboard/GeoAnalysisDashboard";
import type { ObservationResponse, QuerySubmissionResponse } from "@/lib/satquery-api";

export default function AnalysisPage() {
  const { id } = useParams<{ id: string }>();
  const location = useLocation();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const state = location.state as {
    query?: string;
    file?: {
      name: string;
      size: number;
    };
    observation?: ObservationResponse;
  } | null;

  if (!id?.startsWith("ana_")) return <Navigate to="/" replace />;

  const handleAnalysisCreated = (submission: QuerySubmissionResponse) => {
    navigate(`/analysis/${submission.analysis_id}?job=${submission.job_id}`);
  };

  return (
    <main className="h-screen w-full bg-[#050510] text-white overflow-hidden select-none">
      <GeoAnalysisDashboard
        analysisId={id}
        jobId={searchParams.get("job") || undefined}
        uploadedFileName={state?.file?.name}
        uploadedFileSize={
          state?.file ? `${(state.file.size / (1024 * 1024)).toFixed(1)} MB` : undefined
        }
        initialQuery={state?.query}
        initialObservation={state?.observation}
        onBack={() => navigate("/")}
        onAnalysisCreated={handleAnalysisCreated}
      />
    </main>
  );
}
