import type { ManagedRecipeProtectionState } from './configPageMoMPackagesSupport'
import styles from './ConfigPageManagedRecipeBanner.module.css'

interface ConfigPageManagedRecipeBannerProps {
  recipeName?: string
  state: ManagedRecipeProtectionState
}

const CONTENT: Record<
  ManagedRecipeProtectionState,
  { title: string; description: string; tone: 'managed' | 'warning' | 'danger' }
> = {
  active: {
    title: 'Managed Recipe controls the runtime configuration',
    description:
      'Ordinary configuration editors are read-only while this backward-compatible managed activation is active. Use Custom package lifecycle in Mixture-of-Models → Built-in Models to restore the preserved source runtime.',
    tone: 'managed',
  },
  recovering: {
    title: 'Managed Recipe recovery is in progress',
    description:
      'Ordinary configuration editors are locked while the previous runtime state is restored. Managed activation remains unavailable until recovery completes.',
    tone: 'warning',
  },
  inconsistent: {
    title: 'Managed Recipe requires explicit repair',
    description:
      'Routing and probes fail closed, and ordinary configuration editors remain locked. Use Custom package lifecycle in Mixture-of-Models → Built-in Models to repair or restore the managed activation.',
    tone: 'danger',
  },
}

export default function ConfigPageManagedRecipeBanner({
  recipeName,
  state,
}: ConfigPageManagedRecipeBannerProps) {
  const content = CONTENT[state]
  return (
    <aside
      className={`${styles.banner} ${styles[content.tone]}`}
      role={state === 'inconsistent' ? 'alert' : 'status'}
      aria-live={state === 'inconsistent' ? 'assertive' : 'polite'}
      data-testid="managed-recipe-config-lock"
    >
      <div>
        <span className={styles.eyebrow}>Package-managed configuration</span>
        <h2>{content.title}</h2>
        <p>{content.description}</p>
      </div>
      <div className={styles.identity}>
        <span>Active scope</span>
        <strong>{recipeName || 'Managed Recipe package'}</strong>
        <code>{state}</code>
      </div>
    </aside>
  )
}
