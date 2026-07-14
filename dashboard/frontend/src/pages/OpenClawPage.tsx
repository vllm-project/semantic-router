import React, { useCallback, useEffect, useRef, useState } from 'react'
import styles from './OpenClawPage.module.css'
import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import { canManageOpenClaw } from '../utils/accessControl'
import {
  createLatestOpenClawRequest,
  fetchOpenClawJSON,
  getOpenClawErrorMessage,
  type LatestOpenClawRequest,
} from '../utils/openClawRequestSupport'
import { type OpenClawStatus, type TeamProfile } from './OpenClawPageSupport'
import { ArchitectureTab, DashboardTab, StatusTab, TeamTab, WorkerTab } from './OpenClawPageTabs'
import { OpenClawRequestNotice } from './OpenClawRequestNotice'

type OpenClawTab = 'architecture' | 'dashboard' | 'team' | 'provision' | 'status'

const tabMeta: Array<{ key: OpenClawTab; label: string; icon: React.ReactNode }> = [
  {
    key: 'architecture',
    label: 'Overview',
    icon: (
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M12 2l7 4v8l-7 4-7-4V6l7-4z" />
        <path d="M12 22v-8" />
        <path d="M19 6l-7 4-7-4" />
      </svg>
    ),
  },
  {
    key: 'dashboard',
    label: 'Claw Console',
    icon: (
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <line x1="18" y1="20" x2="18" y2="10" />
        <line x1="12" y1="20" x2="12" y2="4" />
        <line x1="6" y1="20" x2="6" y2="14" />
      </svg>
    ),
  },
  {
    key: 'team',
    label: 'Claw Team',
    icon: (
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" />
        <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
        <path d="M16 3.13a4 4 0 0 1 0 7.75" />
      </svg>
    ),
  },
  {
    key: 'provision',
    label: 'Claw Worker',
    icon: (
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M12 2L2 7l10 5 10-5-10-5z" />
        <path d="M2 17l10 5 10-5" />
        <path d="M2 12l10 5 10-5" />
      </svg>
    ),
  },
  {
    key: 'status',
    label: 'Claw Dashboard',
    icon: (
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
      </svg>
    ),
  },
]

const getTabId = (tab: OpenClawTab) => `openclaw-tab-${tab}`
const getPanelId = (tab: OpenClawTab) => `openclaw-panel-${tab}`

const OpenClawPage: React.FC = () => {
  const { user, isLoading: authLoading } = useAuth()
  const { isReadonly, isLoading: readonlyLoading } = useReadonly()
  const permissionsLoading = authLoading || readonlyLoading
  const canManage = !permissionsLoading && !isReadonly && canManageOpenClaw(user)
  const managementDisabled = !canManage
  const [activeTab, setActiveTab] = useState<OpenClawTab>('architecture')
  const [containers, setContainers] = useState<OpenClawStatus[]>([])
  const [teams, setTeams] = useState<TeamProfile[]>([])
  const [statusLoading, setStatusLoading] = useState(true)
  const [teamsLoading, setTeamsLoading] = useState(true)
  const [statusError, setStatusError] = useState('')
  const [teamsError, setTeamsError] = useState('')
  const statusRequestRef = useRef<LatestOpenClawRequest | null>(null)
  const teamsRequestRef = useRef<LatestOpenClawRequest | null>(null)
  const statusInitializedRef = useRef(false)
  const teamsInitializedRef = useRef(false)
  if (!statusRequestRef.current) statusRequestRef.current = createLatestOpenClawRequest()
  if (!teamsRequestRef.current) teamsRequestRef.current = createLatestOpenClawRequest()

  const fetchStatus = useCallback(async (allowHidden = false) => {
    if (!allowHidden && document.hidden) return
    await statusRequestRef.current?.run(
      (signal) => fetchOpenClawJSON<OpenClawStatus[]>('/api/openclaw/status', {}, signal),
      {
        onStart: () => {
          if (!statusInitializedRef.current) setStatusLoading(true)
        },
        onSuccess: (data) => {
          setContainers(Array.isArray(data) ? data : [])
          setStatusError('')
          statusInitializedRef.current = true
        },
        onError: (error) => {
          setStatusError(getOpenClawErrorMessage(error, 'Failed to load OpenClaw status.'))
        },
        onFinish: () => setStatusLoading(false),
      },
    )
  }, [])

  const fetchTeams = useCallback(async (allowHidden = false) => {
    if (!allowHidden && document.hidden) return
    await teamsRequestRef.current?.run(
      (signal) => fetchOpenClawJSON<TeamProfile[]>('/api/openclaw/teams', {}, signal),
      {
        onStart: () => {
          if (!teamsInitializedRef.current) setTeamsLoading(true)
        },
        onSuccess: (data) => {
          setTeams(Array.isArray(data) ? data : [])
          setTeamsError('')
          teamsInitializedRef.current = true
        },
        onError: (error) => {
          setTeamsError(getOpenClawErrorMessage(error, 'Failed to load OpenClaw teams.'))
        },
        onFinish: () => setTeamsLoading(false),
      },
    )
  }, [])

  const refreshAll = useCallback(
    (allowHidden = false) => {
      void Promise.all([fetchStatus(allowHidden), fetchTeams(allowHidden)])
    },
    [fetchStatus, fetchTeams],
  )

  const refreshTeams = useCallback(() => {
    void fetchTeams(true)
  }, [fetchTeams])

  useEffect(() => {
    const onVisibilityChange = () => {
      if (!document.hidden) refreshAll()
    }

    refreshAll(true)
    const interval = window.setInterval(() => refreshAll(), 15000)
    document.addEventListener('visibilitychange', onVisibilityChange)
    return () => {
      window.clearInterval(interval)
      document.removeEventListener('visibilitychange', onVisibilityChange)
      statusRequestRef.current?.cancel()
      teamsRequestRef.current?.cancel()
    }
  }, [refreshAll])

  const focusTabAt = (index: number) => {
    const nextTab = tabMeta[(index + tabMeta.length) % tabMeta.length]
    setActiveTab(nextTab.key)
    window.requestAnimationFrame(() => document.getElementById(getTabId(nextTab.key))?.focus())
  }

  const handleTabKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>, index: number) => {
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
      event.preventDefault()
      focusTabAt(index + 1)
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
      event.preventDefault()
      focusTabAt(index - 1)
    } else if (event.key === 'Home') {
      event.preventDefault()
      focusTabAt(0)
    } else if (event.key === 'End') {
      event.preventDefault()
      focusTabAt(tabMeta.length - 1)
    }
  }

  const runningCount = containers.filter((container) => container.running).length
  const teamCount = teams.length

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerBody}>
          <div className={styles.headerBadgeRow}>
            <span className={`${styles.titleBadge} ${styles.badgePowered}`}>vLLM-SR Powered</span>
            <span className={`${styles.titleBadge} ${styles.badgeTeam}`}>{teamCount} Teams</span>
            <span
              className={`${styles.titleBadge} ${runningCount > 0 ? styles.badgeRunning : styles.badgeStopped}`}
            >
              {runningCount} Running
            </span>
          </div>
          <h1 className={styles.title}>
            <span className={styles.titleLinePrimary}>
              <span className={styles.titleLead}>Semantic Router</span> Powered
            </span>
            <span className={styles.titleLineSecondary}>ClawOS</span>
          </h1>
          <p className={styles.subtitle}>
            Evolved from vLLM-SR built on Semantic Router with System Intelligence.
          </p>
        </div>
        <div className={styles.logoPanel}>
          <img className={styles.logo} src="/openclaw.png" alt="OpenClaw logo" />
        </div>
      </div>

      {!permissionsLoading && !canManage ? (
        <div className={styles.readOnlyNotice} role="status">
          <strong>View-only access.</strong>{' '}
          {isReadonly ? (
            <>This dashboard deployment is in read-only mode, so OpenClaw changes are disabled.</>
          ) : (
            <>
              OpenClaw topology, teams, workers, and runtime status remain visible. The{' '}
              <code>openclaw.manage</code> permission is required for lifecycle or configuration
              changes.
            </>
          )}
        </div>
      ) : null}

      {statusError && activeTab !== 'status' && activeTab !== 'provision' ? (
        <OpenClawRequestNotice
          title="OpenClaw runtime status is unavailable"
          message={statusError}
          onRetry={() => void fetchStatus(true)}
        />
      ) : null}
      {teamsError && activeTab !== 'team' ? (
        <OpenClawRequestNotice
          title="OpenClaw teams are unavailable"
          message={teamsError}
          onRetry={() => void fetchTeams(true)}
        />
      ) : null}

      <div className={styles.tabsBar}>
        <div className={styles.tabs} role="tablist" aria-label="OpenClaw workspace views">
          {tabMeta.map((tab, index) => (
            <button
              key={tab.key}
              id={getTabId(tab.key)}
              type="button"
              role="tab"
              aria-selected={activeTab === tab.key}
              aria-controls={getPanelId(tab.key)}
              tabIndex={activeTab === tab.key ? 0 : -1}
              className={`${styles.tab} ${activeTab === tab.key ? styles.tabActive : ''}`}
              onClick={() => setActiveTab(tab.key)}
              onKeyDown={(event) => handleTabKeyDown(event, index)}
            >
              <span className={styles.tabIcon}>{tab.icon}</span>
              {tab.label}
              {tab.key === 'team' && ` (${teamCount})`}
              {(tab.key === 'provision' || tab.key === 'status') && ` (${containers.length})`}
            </button>
          ))}
        </div>
      </div>

      {activeTab === 'architecture' && (
        <div
          id={getPanelId('architecture')}
          className={styles.tabContentShell}
          role="tabpanel"
          aria-labelledby={getTabId('architecture')}
          tabIndex={0}
        >
          <ArchitectureTab containers={containers} />
        </div>
      )}
      {activeTab === 'dashboard' && (
        <div
          id={getPanelId('dashboard')}
          className={styles.tabContentShell}
          role="tabpanel"
          aria-labelledby={getTabId('dashboard')}
          tabIndex={0}
        >
          <DashboardTab
            containers={containers}
            teams={teams}
            loading={statusLoading || teamsLoading}
            error={statusError || teamsError}
            onRetry={() => refreshAll(true)}
            onSwitchToStatus={() => setActiveTab('status')}
            readOnly={managementDisabled}
          />
        </div>
      )}
      {activeTab === 'team' && (
        <div
          id={getPanelId('team')}
          className={styles.tabContentShell}
          role="tabpanel"
          aria-labelledby={getTabId('team')}
          tabIndex={0}
        >
          <TeamTab
            teams={teams}
            teamsLoading={teamsLoading}
            teamsError={teamsError}
            containers={containers}
            onTeamsUpdated={refreshTeams}
            onRetryTeams={refreshTeams}
            readOnly={managementDisabled}
          />
        </div>
      )}
      {activeTab === 'provision' && (
        <div
          id={getPanelId('provision')}
          className={styles.tabContentShell}
          role="tabpanel"
          aria-labelledby={getTabId('provision')}
          tabIndex={0}
        >
          <WorkerTab
            containers={containers}
            teams={teams}
            workersLoading={statusLoading}
            workersError={statusError}
            onProvisioned={() => refreshAll(true)}
            onRetryWorkers={() => void fetchStatus(true)}
            onSwitchToTeam={() => setActiveTab('team')}
            onSwitchToStatus={() => setActiveTab('status')}
            readOnly={managementDisabled}
          />
        </div>
      )}
      {activeTab === 'status' && (
        <div
          id={getPanelId('status')}
          className={styles.tabContentShell}
          role="tabpanel"
          aria-labelledby={getTabId('status')}
          tabIndex={0}
        >
          <StatusTab
            containers={containers}
            statusLoading={statusLoading}
            statusError={statusError}
            onRefresh={() => refreshAll(true)}
            readOnly={managementDisabled}
          />
        </div>
      )}
    </div>
  )
}

export default OpenClawPage
