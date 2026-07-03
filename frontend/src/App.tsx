import { useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from './store/useAppStore'
import {
  useHealthPoll,
  useTrainingWs,
  fetchEnvs,
  fetchEnvSkill,
  fetchHighScores,
  fetchPlayScores,
  fetchSystem,
} from './api/client'
import TopBar from './components/TopBar'
import Sidebar from './components/Sidebar'
import EnvPreview from './components/EnvPreview'
import RewardChart from './components/RewardChart'
import BottomPanels from './components/BottomPanels'
import PlayScoreGate from './components/PlayScoreGate'
import AnalysisSurface from './components/analysis/AnalysisSurface'

export default function App() {
  const { locale, theme, backendStatus, setEnvs, setSelectedEnvId, setHighScores } = useAppStore()
  const selectedEnvId = useAppStore((s) => s.selectedEnvId)
  const setEnvSkill   = useAppStore((s) => s.setEnvSkill)
  const setPlayScores = useAppStore((s) => s.setPlayScores)
  const setGpuAvailable = useAppStore((s) => s.setGpuAvailable)
  const { i18n } = useTranslation()

  useEffect(() => {
    void i18n.changeLanguage(locale)
  }, [locale, i18n])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  // Fetch env catalog once backend is up; re-fetch on reconnect (data is static)
  useEffect(() => {
    if (backendStatus !== 'online') return
    void fetchEnvs()
      .then((data) => {
        setEnvs(data)
        const { selectedEnvId } = useAppStore.getState()
        if (data.length > 0 && selectedEnvId === null) setSelectedEnvId(data[0].id)
      })
      .catch(() => {})
  }, [backendStatus, setEnvs, setSelectedEnvId])

  // Load persisted all-time high scores once backend is up (live updates arrive via WS).
  useEffect(() => {
    if (backendStatus !== 'online') return
    void fetchHighScores().then(setHighScores).catch(() => {})
  }, [backendStatus, setHighScores])

  // Detect GPU availability once backend is up — gates GPU-only training (Atari) in the UI (G4a).
  useEffect(() => {
    if (backendStatus !== 'online') return
    void fetchSystem().then((s) => setGpuAvailable(s.gpu_available)).catch(() => {})
  }, [backendStatus, setGpuAvailable])

  // Skill-band thresholds for the selected env (single source for the skill meter + play rating).
  useEffect(() => {
    if (backendStatus !== 'online' || !selectedEnvId) return
    void fetchEnvSkill(selectedEnvId).then(setEnvSkill).catch(() => setEnvSkill(null))
  }, [backendStatus, selectedEnvId, setEnvSkill])

  // Play leaderboards (Human + AI) for the selected env — feeds the bottom boards + meter markers.
  useEffect(() => {
    if (backendStatus !== 'online' || !selectedEnvId) return
    void fetchPlayScores(selectedEnvId).then(setPlayScores).catch(() => setPlayScores(null))
  }, [backendStatus, selectedEnvId, setPlayScores])

  useHealthPoll()
  useTrainingWs()

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      <TopBar />
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <Sidebar />
        <div style={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
          <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
            <EnvPreview />
            <RewardChart />
          </div>
          <BottomPanels />
        </div>
      </div>
      {/* Invisible: watches play results, auto-records AI scores, prompts a human for a name. */}
      <PlayScoreGate />
      {/* X6: fullscreen DataLab research surface — a portal over the dashboard (self-gates on analysisOpen). */}
      <AnalysisSurface />
    </div>
  )
}
