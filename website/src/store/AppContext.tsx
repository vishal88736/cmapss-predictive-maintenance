import React, { createContext, useCallback, useContext, useMemo, useState } from 'react';

export type View = 'overview' | 'fleet' | 'twin' | 'telemetry' | 'predictions' | 'alerts' | 'models';

interface AppState {
  view: View;
  setView: (v: View) => void;
  selectedEngineId: number | null;
  setSelectedEngineId: (id: number | null) => void;
  twinEngineId: number;
  setTwinEngineId: (id: number) => void;
  openTwin: (id: number) => void;
  scrubCycle: number | null;
  setScrubCycle: (c: number | null) => void;
  primarySensor: string;
  setPrimarySensor: (s: string) => void;
  pinned: string[];
  togglePin: (s: string) => void;
  compareA: number;
  compareB: number;
  setCompareA: (n: number) => void;
  setCompareB: (n: number) => void;
  commandOpen: boolean;
  setCommandOpen: (b: boolean) => void;
  helpOpen: boolean;
  setHelpOpen: (b: boolean) => void;
  focusMode: boolean;
  setFocusMode: (b: boolean) => void;
  presentationMode: boolean;
  setPresentationMode: (b: boolean) => void;
  sysOpen: boolean;
  setSysOpen: (b: boolean) => void;
  fleetQuery: string;
  setFleetQuery: (s: string) => void;
  fleetStatus: string;
  setFleetStatus: (s: string) => void;
  aiResponse: string | null;
  setAiResponse: (s: string | null) => void;
}

const Ctx = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [view, setView] = useState<View>('overview');
  const [selectedEngineId, setSelectedEngineId] = useState<number | null>(null);
  const [twinEngineId, setTwinEngineId] = useState(24);
  const [scrubCycle, setScrubCycle] = useState<number | null>(null);
  const [primarySensor, setPrimarySensor] = useState('T24');
  const [pinned, setPinned] = useState<string[]>(['P30']);
  const [compareA, setCompareA] = useState(24);
  const [compareB, setCompareB] = useState(71);
  const [commandOpen, setCommandOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [focusMode, setFocusMode] = useState(false);
  const [presentationMode, setPresentationMode] = useState(false);
  const [sysOpen, setSysOpen] = useState(false);
  const [fleetQuery, setFleetQuery] = useState('');
  const [fleetStatus, setFleetStatus] = useState('ALL');
  const [aiResponse, setAiResponse] = useState<string | null>(null);

  const togglePin = useCallback((s: string) => {
    setPinned((p) => (p.includes(s) ? p.filter((x) => x !== s) : [...p, s].slice(0, 3)));
  }, []);

  const openTwin = useCallback((id: number) => {
    setTwinEngineId(id);
    setScrubCycle(null);
    setView('twin');
    setSelectedEngineId(null);
  }, []);

  const value = useMemo(
    () => ({
      view, setView,
      selectedEngineId, setSelectedEngineId,
      twinEngineId, setTwinEngineId, openTwin,
      scrubCycle, setScrubCycle,
      primarySensor, setPrimarySensor,
      pinned, togglePin,
      compareA, compareB, setCompareA, setCompareB,
      commandOpen, setCommandOpen,
      helpOpen, setHelpOpen,
      focusMode, setFocusMode,
      presentationMode, setPresentationMode,
      sysOpen, setSysOpen,
      fleetQuery, setFleetQuery,
      fleetStatus, setFleetStatus,
      aiResponse, setAiResponse,
    }),
    [view, selectedEngineId, twinEngineId, scrubCycle, primarySensor, pinned, compareA, compareB, commandOpen, helpOpen, focusMode, presentationMode, sysOpen, fleetQuery, fleetStatus, aiResponse, openTwin, togglePin],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error('useApp outside provider');
  return v;
}
