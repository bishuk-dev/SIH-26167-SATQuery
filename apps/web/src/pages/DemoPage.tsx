import React, { useState } from "react";
import RuixenMoonChat, { type AnalysisRequest } from "@/components/ui/ruixen-moon-chat";
import GeoAnalysisDashboard from "@/components/dashboard/GeoAnalysisDashboard";

export default function DemoPage() {
  const [currentView, setCurrentView] = useState<"chat" | "dashboard">("chat");
  const [analysisData, setAnalysisData] = useState<AnalysisRequest | null>(null);

  const handleStartAnalysis = (request: AnalysisRequest) => {
    setAnalysisData(request);
    setCurrentView("dashboard");
  };

  const handleBackToChat = () => {
    setCurrentView("chat");
  };

  return (
    <main className="h-screen w-full bg-[#050510] text-white overflow-hidden select-none">
      {currentView === "chat" ? (
        <RuixenMoonChat onStartAnalysis={handleStartAnalysis} />
      ) : (
        <GeoAnalysisDashboard
          uploadedFileName={analysisData?.file.name}
          uploadedFileSize={analysisData?.file.size}
          initialQuery={analysisData?.query}
          onBack={handleBackToChat}
        />
      )}
    </main>
  );
}
