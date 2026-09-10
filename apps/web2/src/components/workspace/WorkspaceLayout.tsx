import React, { useState } from 'react';
import { WorkspaceHeader } from './WorkspaceHeader';
import { WorkspaceFooter } from './WorkspaceFooter';
import { LeftPanel } from './LeftPanel';
import { RightPanel } from './RightPanel';
import { CenterMap } from './CenterMap';

export const WorkspaceLayout = () => {
  const [opticalOpacity, setOpticalOpacity] = useState(100);
  const [diffActive, setDiffActive] = useState(true);
  const [activeTarget, setActiveTarget] = useState<string | null>(null);
  const [observationId, setObservationId] = useState<string>('example');

  const handleInspect = (targetId: string) => {
    setActiveTarget(targetId);
    setTimeout(() => {
      setActiveTarget(null);
    }, 1200);
  };

  return (
    <div className="bg-[#080e1a] text-slate-200 select-none overflow-hidden h-screen w-screen flex flex-col antialiased font-body-sm absolute inset-0 z-50">
      <WorkspaceHeader />
      
      <main className="flex-1 flex overflow-hidden relative">
        <LeftPanel onInspect={handleInspect} observationId={observationId} />
        <CenterMap 
          opticalOpacity={opticalOpacity} 
          diffActive={diffActive} 
          activeTarget={activeTarget}
        />
        <RightPanel 
          opticalOpacity={opticalOpacity} 
          setOpticalOpacity={setOpticalOpacity} 
          diffActive={diffActive} 
          toggleDiff={() => setDiffActive(!diffActive)} 
          setObservationId={setObservationId}
        />
      </main>

      <WorkspaceFooter />
    </div>
  );
};
