import { create } from 'zustand'
import type { Alert } from '@/types/alert'

export interface Toast {
  id: string
  title: string
  description?: string
  tone?: 'info' | 'ok' | 'warn' | 'crit'
  actionLabel?: string
  action?: () => void
  duration?: number
}

interface AppState {
  paletteOpen: boolean
  helpOpen: boolean
  settingsOpen: boolean
  explorerEngineId: string | null
  lastEngineId: string | null
  focusMode: boolean
  presentation: { active: boolean; engineId: string | null }
  compareIds: [string, string]
  pinnedSensors: string[]
  primarySensor: string
  scrubCycle: Record<string, number>
  alerts: Alert[]
  toasts: Toast[]
  liveFeed: boolean
  tempUnit: 'R' | 'K'
  lastSync: number
  setPaletteOpen: (open: boolean) => void
  setHelpOpen: (open: boolean) => void
  setSettingsOpen: (open: boolean) => void
  openExplorer: (engineId: string) => void
  closeExplorer: () => void
  toggleFocusMode: () => void
  setFocusMode: (v: boolean) => void
  startPresentation: (engineId: string) => void
  stopPresentation: () => void
  setCompareIds: (ids: [string, string]) => void
  togglePin: (sensorId: string) => void
  setPrimarySensor: (sensorId: string) => void
  setScrub: (engineId: string, cycle: number) => void
  pushAlert: (alert: Alert) => void
  pushToast: (toast: Omit<Toast, 'id'>) => void
  dismissToast: (id: string) => void
  setLiveFeed: (v: boolean) => void
  setTempUnit: (u: 'R' | 'K') => void
  tickSync: () => void
}

const persisted = JSON.parse(localStorage.getItem('aeroguard-settings') || '{}') as Partial<AppState>

export const useAppStore = create<AppState>((set, get) => ({
  paletteOpen: false,
  helpOpen: false,
  settingsOpen: false,
  explorerEngineId: null,
  lastEngineId: '024',
  focusMode: false,
  presentation: { active: false, engineId: null },
  compareIds: ['024', '071'],
  pinnedSensors: ['Ps30'],
  primarySensor: 'T24',
  scrubCycle: {},
  alerts: [],
  toasts: [],
  liveFeed: persisted.liveFeed ?? true,
  tempUnit: persisted.tempUnit ?? 'R',
  lastSync: Date.now(),
  setPaletteOpen: (open) => set({ paletteOpen: open }),
  setHelpOpen: (open) => set({ helpOpen: open }),
  setSettingsOpen: (open) => set({ settingsOpen: open }),
  openExplorer: (engineId) => set({ explorerEngineId: engineId, lastEngineId: engineId }),
  closeExplorer: () => set({ explorerEngineId: null }),
  toggleFocusMode: () => set((s) => ({ focusMode: !s.focusMode })),
  setFocusMode: (v) => set({ focusMode: v }),
  startPresentation: (engineId) => set({ presentation: { active: true, engineId }, paletteOpen: false }),
  stopPresentation: () => set({ presentation: { active: false, engineId: null } }),
  setCompareIds: (ids) => set({ compareIds: ids }),
  togglePin: (sensorId) =>
    set((s) => ({
      pinnedSensors: s.pinnedSensors.includes(sensorId) ? s.pinnedSensors.filter((x) => x !== sensorId) : [...s.pinnedSensors, sensorId]
    })),
  setPrimarySensor: (sensorId) =>
    set((s) => ({
      primarySensor: sensorId,
      pinnedSensors: sensorId !== s.primarySensor && s.pinnedSensors.includes(sensorId) ? s.pinnedSensors.filter((x) => x !== sensorId) : s.pinnedSensors
    })),
  setScrub: (engineId, cycle) => set((s) => ({ scrubCycle: { ...s.scrubCycle, [engineId]: cycle } })),
  pushAlert: (alert) => set((s) => ({ alerts: [alert, ...s.alerts].slice(0, 80) })),
  pushToast: (toast) => {
    const id = Math.random().toString(36).slice(2, 10)
    set((s) => ({ toasts: [...s.toasts, { ...toast, id }] }))
    const dur = toast.duration ?? 5200
    setTimeout(() => get().dismissToast(id), dur)
  },
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
  setLiveFeed: (v) => {
    set({ liveFeed: v })
    persistSettings(get())
  },
  setTempUnit: (u) => {
    set({ tempUnit: u })
    persistSettings(get())
  },
  tickSync: () => set({ lastSync: Date.now() })
}))

function persistSettings(s: AppState) {
  localStorage.setItem('aeroguard-settings', JSON.stringify({ liveFeed: s.liveFeed, tempUnit: s.tempUnit }))
}
