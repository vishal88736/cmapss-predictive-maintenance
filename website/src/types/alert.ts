export type AlertSeverity = 'critical' | 'warning' | 'info'

export interface Alert {
  id: string
  ts: number
  severity: AlertSeverity
  engineId: string
  title: string
  detail: string
  cycle?: number
  sensorId?: string
  live?: boolean
  acknowledged?: boolean
}
