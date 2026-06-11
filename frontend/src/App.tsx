import { useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from './store/useAppStore'
import { useHealthPoll } from './api/client'
import TopBar from './components/TopBar'
import Sidebar from './components/Sidebar'
import EnvPreview from './components/EnvPreview'
import RewardChart from './components/RewardChart'
import BottomPanels from './components/BottomPanels'

export default function App() {
  const { locale, theme } = useAppStore()
  const { i18n } = useTranslation()

  useEffect(() => {
    void i18n.changeLanguage(locale)
  }, [locale, i18n])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  useHealthPoll()

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
