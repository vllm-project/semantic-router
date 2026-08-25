import type { ReactNode } from 'react'
import ProductIcon, { type ProductIconName } from '../components/ProductIcon'
import type { AccessOverview } from '../utils/inferenceAccessApi'
import type { AccessView } from './AccessControlPageSupport'
import styles from './AccessControlPage.module.css'

interface AccessControlWorkspaceProps {
  activeView: AccessView
  activeMeta: {
    section: string
    label: string
    description: string
  }
  visibleNavItems: Array<{
    id: AccessView
    icon: ProductIconName
    label: string
  }>
  overview: AccessOverview
  liveState: 'checking' | 'live' | 'error'
  canInvite: boolean
  canCreate: boolean
  createLabel: string
  error: string
  loading: boolean
  toast: string
  onCheck: () => void
  onNavigate: (view: AccessView) => void
  onInvite: () => void
  onCreate: () => void
  onDismissError: () => void
  children: ReactNode
}

const AccessControlWorkspace = ({
  activeView,
  activeMeta,
  visibleNavItems,
  overview,
  liveState,
  canInvite,
  canCreate,
  createLabel,
  error,
  loading,
  toast,
  onCheck,
  onNavigate,
  onInvite,
  onCreate,
  onDismissError,
  children,
}: AccessControlWorkspaceProps) => (
  <>
    <header className={`${styles.hero} ${styles.heroCompact}`}>
      <div className={styles.heroCopy}>
        <div className={styles.heroTopline}>
          <span className={styles.eyebrow}>Access Control</span>
          <span className={styles.heroBrand}>
            <img src="/vllm.png" alt="" />
            vllm-sr
          </span>
        </div>
        <h1>Every model. The right audience.</h1>
        <p>Give users and teams exactly the models and capacity they need.</p>
      </div>
      <div className={styles.heroPulse}>
        <button
          type="button"
          className={`${styles.liveButton} ${styles[`live${liveState}`]}`}
          onClick={onCheck}
          aria-label="Check access-control service"
        >
          <span /> {liveState === 'checking' ? 'Checking' : liveState === 'live' ? 'Live' : 'Retry'}
        </button>
        <div>
          <strong>{overview.requestsToday.toLocaleString('en-US')}</strong>
          <span>requests today</span>
        </div>
        <div>
          <strong>{overview.tokensToday.toLocaleString('en-US')}</strong>
          <span>tokens today</span>
        </div>
      </div>
    </header>

    <nav className={styles.sectionNav} aria-label="Access control">
      {visibleNavItems.map((item) => (
        <button
          type="button"
          key={item.id}
          className={activeView === item.id ? styles.sectionNavActive : ''}
          onClick={() => onNavigate(item.id)}
          aria-current={activeView === item.id ? 'page' : undefined}
        >
          <ProductIcon name={item.icon} />
          {item.label}
        </button>
      ))}
    </nav>

    <main className={styles.surface}>
      <div className={styles.surfaceHeader}>
        <div>
          <span>{activeMeta.section}</span>
          <h2>{activeMeta.label}</h2>
          <p>{activeMeta.description}</p>
        </div>
        <div className={styles.headerActions}>
          {canInvite ? (
            <button type="button" className={styles.primaryButton} onClick={onInvite}>
              <ProductIcon name="plus" /> Invite user
            </button>
          ) : null}
          {canCreate ? (
            <button type="button" className={styles.primaryButton} onClick={onCreate}>
              <ProductIcon name="plus" /> {createLabel}
            </button>
          ) : null}
        </div>
      </div>

      {error ? (
        <div className={styles.inlineError} role="alert">
          {error}
          <button type="button" onClick={onDismissError}>
            <ProductIcon name="close" /> Dismiss
          </button>
        </div>
      ) : null}
      {loading ? (
        <div className={styles.skeletonGrid}>
          <i />
          <i />
          <i />
          <i />
        </div>
      ) : null}
      {!loading ? children : null}
    </main>

    {toast ? (
      <div className={styles.toast} role="status">
        <ProductIcon name="check" />
        {toast}
      </div>
    ) : null}
  </>
)

export default AccessControlWorkspace
