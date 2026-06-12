import { useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from './store/useAppStore'
import { useHealthPoll, useTrainingWs, fetchEnvs, fetchHighScores } from './api/client'
import TopBar from './components/TopBar'
import Sidebar from './components/Sidebar'
import EnvPreview from './components/EnvPreview'
import RewardChart from './components/RewardChart'
import BottomPanels from './components/BottomPanels'

export default function App() {
  const { locale, theme, backendStatus, setEnvs, setSelectedEnvId, setHighScores } = useAppStore()
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
    </div>
  )
}
