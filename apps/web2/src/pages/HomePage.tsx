import React from "react";
import { useNavigate } from "react-router-dom";
import RuixenMoonChat, { type AnalysisRequest } from "@/components/ui/ruixen-moon-chat";

export default function HomePage() {
  const navigate = useNavigate();

  const handleStartAnalysis = (request: AnalysisRequest) => {
    let analysisId = "sq-barcelona-01";
    if (request.file.name && request.file.name !== "barcelona_port_multisensor.tif") {
      const sanitized = request.file.name
        .replace(/\.[^/.]+$/, "")
        .replace(/[^a-zA-Z0-9]/g, "-")
        .toLowerCase()
        .replace(/-+/g, "-")
        .slice(0, 24);
      analysisId = `sq-${sanitized || Date.now().toString(36)}`;
    }

    navigate(`/analysis/${analysisId}`, {
      state: {
        query: request.query,
        file: request.file,
      },
    });
  };

  return (
    <main className="h-screen w-full bg-[#050510] text-white overflow-hidden select-none">
      <RuixenMoonChat onStartAnalysis={handleStartAnalysis} />
    </main>
  );
}
