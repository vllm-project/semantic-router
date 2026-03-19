import React, { useCallback, useEffect, useState } from 'react'
import styles from './OpenClawPage.module.css'
import { useReadonly } from '../contexts/ReadonlyContext'
import {
  type OpenClawStatus,
  type TeamProfile,
} from './OpenClawPageSupport'
import {
  ArchitectureTab,
  DashboardTab,
  StatusTab,
  TeamTab,
  WorkerTab,
} from './OpenClawPageTabs'

type OpenClawTab = 'architecture' | 'dashboard' | 'team' | 'provision' | 'status'

const tabMeta: Array<{ key: OpenClawTab; label: string; icon: React.ReactNode }> = [
  {
    key: 'architecture',
    label: 'Overview',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
      </svg>
    ),
  },
]

const OpenClawPage: React.FC = () => {
  const { isReadonly, isLoading: readonlyLoading } = useReadonly()
  const managementDisabled = readonlyLoading || isReadonly
  const [activeTab, setActiveTab] = useState<OpenClawTab>('architecture')
  const [containers, setContainers] = useState<OpenClawStatus[]>([])
  const [teams, setTeams] = useState<TeamProfile[]>([])
  const [statusLoading, setStatusLoading] = useState(true)
  const [teamsLoading, setTeamsLoading] = useState(true)

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/openclaw/status')
      if (res.ok) {
        const data = await res.json()
        setContainers(Array.isArray(data) ? data : [])
      }
    } catch {
      // ignore
    } finally {
      setStatusLoading(false)
    }
  }, [])

  const fetchTeams = useCallback(async () => {
    try {
      const res = await fetch('/api/openclaw/teams')
      if (res.ok) {
        const data = await res.json()
        setTeams(Array.isArray(data) ? data : [])
      } else {
        setTeams([])
      }
    } catch {
      setTeams([])
    } finally {
      setTeamsLoading(false)
    }
  }, [])

  const refreshAll = useCallback(() => {
    void Promise.all([fetchStatus(), fetchTeams()])
  }, [fetchStatus, fetchTeams])

  useEffect(() => {
    refreshAll()
    const interval = setInterval(refreshAll, 15000)
    return () => clearInterval(interval)
  }, [refreshAll])

  const runningCount = containers.filter(container => container.running).length
  const teamCount = teams.length

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerBody}>
          <div className={styles.headerBadgeRow}>
            <span className={`${styles.titleBadge} ${styles.badgePowered}`}>vLLM-SR Powered</span>
            <span className={`${styles.titleBadge} ${styles.badgeTeam}`}>{teamCount} Teams</span>
            <span className={`${styles.titleBadge} ${runningCount > 0 ? styles.badgeRunning : styles.badgeStopped}`}>
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

      <div className={styles.tabsBar}>
        <div className={styles.tabs}>
          {tabMeta.map(tab => (
            <button
              key={tab.key}
              className={`${styles.tab} ${activeTab === tab.key ? styles.tabActive : ''}`}
              onClick={() => setActiveTab(tab.key)}
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
        <div className={styles.tabContentShell}>
          <ArchitectureTab containers={containers} />
        </div>
      )}
      {activeTab === 'dashboard' && (
        <div className={styles.tabContentShell}>
          <DashboardTab containers={containers} teams={teams} onSwitchToStatus={() => setActiveTab('status')} />
        </div>
      )}
      {activeTab === 'team' && (
        <div className={styles.tabContentShell}>
          <TeamTab
            teams={teams}
            teamsLoading={teamsLoading}
            containers={containers}
            onTeamsUpdated={fetchTeams}
            readOnly={managementDisabled}
          />
        </div>
      )}
      {activeTab === 'provision' && (
        <div className={styles.tabContentShell}>
          <WorkerTab
            containers={containers}
            teams={teams}
            onProvisioned={refreshAll}
            onSwitchToTeam={() => setActiveTab('team')}
            onSwitchToStatus={() => setActiveTab('status')}
            readOnly={managementDisabled}
          />
        </div>
      )}
      {activeTab === 'status' && (
        <div className={styles.tabContentShell}>
          <StatusTab
            containers={containers}
            statusLoading={statusLoading}
            onRefresh={refreshAll}
            readOnly={managementDisabled}
          />
        </div>
      )}
    </div>
  )
}

export default OpenClawPage
