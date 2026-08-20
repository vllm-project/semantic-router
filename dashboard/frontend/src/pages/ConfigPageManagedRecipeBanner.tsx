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
    title: 'Recipe package active',
    description: 'Deactivate the package to edit this configuration.',
    tone: 'managed',
  },
  recovering: {
    title: 'Restoring configuration',
    description: 'Editing will return when the restore is complete.',
    tone: 'warning',
  },
  inconsistent: {
    title: 'Recipe package needs attention',
    description: 'Repair or deactivate the package before editing.',
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
        <span className={styles.eyebrow}>Managed configuration</span>
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
