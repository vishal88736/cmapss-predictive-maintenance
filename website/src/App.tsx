import { Routes, Route, Navigate } from 'react-router-dom'
import AppShell from '@/components/shell/AppShell'
import Overview from '@/pages/Overview'
import Fleet from '@/pages/Fleet'
import Engines from '@/pages/Engines'
import TelemetryPage from '@/pages/TelemetryPage'
import Predictions from '@/pages/Predictions'
import Alerts from '@/pages/Alerts'
import Models from '@/pages/Models'
import EngineTwin from '@/pages/EngineTwin'
import ComparePage from '@/pages/ComparePage'

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<Overview />} />
        <Route path="/fleet" element={<Fleet />} />
        <Route path="/engines" element={<Engines />} />
        <Route path="/telemetry" element={<TelemetryPage />} />
        <Route path="/predictions" element={<Predictions />} />
        <Route path="/alerts" element={<Alerts />} />
        <Route path="/models" element={<Models />} />
        <Route path="/engine/:id" element={<EngineTwin />} />
        <Route path="/compare/:a/:b" element={<ComparePage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
