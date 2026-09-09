import React from "react";
import { useParams, useLocation, useNavigate } from "react-router-dom";
import GeoAnalysisDashboard from "@/components/dashboard/GeoAnalysisDashboard";

export default function AnalysisPage() {
  const { id } = useParams<{ id: string }>();
  const location = useLocation();
  const navigate = useNavigate();

  const state = location.state as {
    query?: string;
    file?: {
      name: string;
      size: string;
    };
  } | null;

  const defaultFileName =
    id === "sq-barcelona-01"
      ? "barcelona_port_multisensor.tif"
      : `${id?.replace(/^sq-/, "") || "raster"}_multisensor.tif`;

  return (
    <main className="h-screen w-full bg-[#050510] text-white overflow-hidden select-none">
      <GeoAnalysisDashboard
        analysisId={id}
        uploadedFileName={state?.file?.name || defaultFileName}
        uploadedFileSize={state?.file?.size || "48.2 MB"}
        initialQuery={
          state?.query ||
          "Assess coastal port infrastructure changes between T0 and T1, and identify unpermitted shoreline backfill."
        }
        onBack={() => navigate("/")}
      />
    </main>
  );
}
