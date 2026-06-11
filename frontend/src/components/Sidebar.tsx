import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'

export default function Sidebar() {
  const { t } = useTranslation()
  const envs            = useAppStore((s) => s.envs)
  const selectedEnvId   = useAppStore((s) => s.selectedEnvId)
  const locale          = useAppStore((s) => s.locale)
  const setSelectedEnvId = useAppStore((s) => s.setSelectedEnvId)

  return (
    <aside style={{
      width: 260, flexShrink: 0,
      background: 'var(--surface)', borderRight: '1px solid var(--border)',
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
    }}>
      <div style={{
        padding: '10px 14px', borderBottom: '1px solid var(--border)',
        fontWeight: 600, fontSize: 13, color: 'var(--text-h)',
      }}>
        {t('sidebar.title')}
      </div>

      {/* Game selector */}
      <div style={{ padding: '10px 14px', borderBottom: '1px solid var(--border)' }}>
        <label style={{
          display: 'block', marginBottom: 4,
          fontSize: 11, color: 'var(--text-muted)',
          textTransform: 'uppercase', letterSpacing: '0.05em',
        }}>
          {t('sidebar.game_selector')}
        </label>
        <select
          value={selectedEnvId ?? ''}
          onChange={(e) => { setSelectedEnvId(e.target.value || null) }}
          disabled={envs.length === 0}
          style={{
            width: '100%', padding: '5px 8px',
            background: 'var(--surface-2)', color: 'var(--text)',
            border: '1px solid var(--border)', borderRadius: 4,
            fontSize: 13, cursor: envs.length === 0 ? 'default' : 'pointer',
          }}
        >
          {envs.length === 0
            ? <option value="">{t('sidebar.loading_envs')}</option>
            : envs.map((env) => (
                <option key={env.id} value={env.id}>
                  {env.display_name[locale]}
                </option>
              ))
          }
        </select>
      </div>

      <div style={{
        flex: 1, padding: 14, display: 'flex', alignItems: 'center',
        justifyContent: 'center', color: 'var(--text-muted)', fontSize: 12,
        textAlign: 'center',
      }}>
        {/* Parameter sliders will appear here in B3 */}
        <span>—</span>
      </div>
    </aside>
  )
}
