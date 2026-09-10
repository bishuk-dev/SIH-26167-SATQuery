import { Routes, Route, useLocation } from 'react-router-dom';
import { ThreeBackground } from './components/ThreeBackground';
import { Header } from './components/Header';
import { HomeFooter } from './components/HomeFooter';
import { MapViewer } from './components/MapViewer';
import { AnalysisDocs } from './pages/AnalysisDocs';
import { SensorCatalog } from './pages/SensorCatalog';
import { ApiKeys } from './pages/ApiKeys';
import { Benchmarks } from './pages/Benchmarks';
import { NewLandingPage } from './pages/NewLandingPage';
import { WorkspaceLayout } from './components/workspace/WorkspaceLayout';
import { AnalysisResponse } from './pages/AnalysisResponse';
import './index.css';

function App() {
  const location = useLocation();
  const isWorkspace = location.pathname === '/workspace';
  const isLandingPage = location.pathname === '/';

  return (
    <div className={`min-h-screen flex flex-col justify-between text-slate-100 font-sans selection:bg-cyan-500/30 selection:text-cyan-200 relative overflow-x-hidden ${isWorkspace ? 'bg-[#080e1a]' : 'bg-space-900'}`}>
      {!isWorkspace && !isLandingPage && <ThreeBackground />}
      {!isWorkspace && !isLandingPage && <MapViewer />}
      
      {!isWorkspace && !isLandingPage && <Header />}
      <Routes>
        <Route path="/" element={<NewLandingPage />} />
        <Route path="/analysis" element={<AnalysisResponse />} />
        <Route path="/workspace" element={<WorkspaceLayout />} />
        <Route path="/docs" element={<AnalysisDocs />} />
        <Route path="/catalog" element={<SensorCatalog />} />
        <Route path="/api-keys" element={<ApiKeys />} />
        <Route path="/benchmarks" element={<Benchmarks />} />
      </Routes>
      {!isWorkspace && !isLandingPage && <HomeFooter />}
    </div>
  );
}

export default App;
