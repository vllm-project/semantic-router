import { useEffect, useRef, useState } from 'react'

import ConfirmDialog from '../components/ConfirmDialog'
import type { RecipeActivationPlan, RecipePackageList, RecipePackageSummary } from '../types/recipe'
import {
  activateRecipePackage,
  deactivateRecipePackage,
  listRecipePackages,
  previewRecipePackageActivation,
  previewRecipePackageDeactivation,
} from '../utils/recipeApi'
import styles from './ConfigPageMoMPackagesPanel.module.css'
import {
  formatPackageInstalledAt,
  presentRecipePackageWarning,
  presentRecipePackageError,
  RECIPE_PACKAGE_PAGE_SIZE,
  type RecipePackageCapabilities,
  type RecipePackageErrorPresentation,
} from './configPageMoMPackagesSupport'

interface ConfigPageMoMPackagesPanelProps {
  packageCapabilities: RecipePackageCapabilities
  recipeRevision: number
  onRecipeLifecycleChanged: () => Promise<boolean>
}

function shortDigest(value: string): string {
  return value.length > 24 ? `${value.slice(0, 24)}…` : value
}

export default function ConfigPageMoMPackagesPanel({
  packageCapabilities,
  recipeRevision,
  onRecipeLifecycleChanged,
}: ConfigPageMoMPackagesPanelProps) {
  const { canActivate, activationBlocker } = packageCapabilities
  const activationUnavailableTitle =
    activationBlocker === 'server_readonly'
      ? 'Server read-only'
      : activationBlocker === 'runtime_config_readonly'
        ? 'Runtime configuration is read-only'
        : activationBlocker === 'permission_required'
          ? 'Requires config.deploy permission'
          : undefined
  const [inventory, setInventory] = useState<RecipePackageList | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<RecipePackageErrorPresentation | null>(null)
  const [listRevision, setListRevision] = useState(0)
  const [page, setPage] = useState(1)
  const [query, setQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [actionError, setActionError] = useState<RecipePackageErrorPresentation | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [activeOperation, setActiveOperation] = useState<string | null>(null)
  const [activationTarget, setActivationTarget] = useState<RecipePackageSummary | null>(null)
  const [activationPlan, setActivationPlan] = useState<RecipeActivationPlan | null>(null)
  const [deactivationTarget, setDeactivationTarget] = useState<RecipePackageSummary | null>(null)
  const [deactivationPlan, setDeactivationPlan] = useState<RecipeActivationPlan | null>(null)
  const actionControllerRef = useRef<AbortController | null>(null)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      actionControllerRef.current?.abort()
    }
  }, [])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const nextQuery = query.trim()
      if (nextQuery === debouncedQuery) return
      setPage(1)
      setDebouncedQuery(nextQuery)
    }, 250)
    return () => window.clearTimeout(timer)
  }, [debouncedQuery, query])

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setLoadError(null)
    void listRecipePackages(
      { page, pageSize: RECIPE_PACKAGE_PAGE_SIZE, query: debouncedQuery },
      controller.signal,
    )
      .then((next) => {
        if (!controller.signal.aborted) setInventory(next)
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setInventory(null)
          setLoadError(presentRecipePackageError(reason, 'load'))
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [debouncedQuery, listRevision, page, recipeRevision])

  useEffect(() => {
    if (inventory && page > Math.max(inventory.total_pages, 1)) {
      setPage(Math.max(inventory.total_pages, 1))
    }
  }, [inventory, page])

  const activationState = inventory?.activation_state ?? 'none'
  const activationBlocked = activationState === 'recovering'
  const repairMode = activationState === 'inconsistent'
  const operationPending = activeOperation !== null

  const handleActivate = async () => {
    const target = activationTarget
    const plan = activationPlan
    if (!target || !plan || operationPending || !canActivate || activationBlocked) return

    const controller = new AbortController()
    actionControllerRef.current?.abort()
    actionControllerRef.current = controller
    setActiveOperation(`activate:${target.recipe_digest}`)
    setActionError(null)
    setSuccess(null)
    try {
      await activateRecipePackage(
        target.recipe_digest,
        target.warnings.length > 0,
        controller.signal,
        {
          expected_plan_digest: plan.plan_digest,
          confirm_stack_recreation: plan.mode === 'stack_recreation',
        },
      )
      if (controller.signal.aborted) return
      const configRefreshed = await onRecipeLifecycleChanged()
      if (controller.signal.aborted) return
      setActivationTarget(null)
      setActivationPlan(null)
      setPage(1)
      if (configRefreshed) {
        setSuccess(
          `${target.name} ${target.version} is active. The Router configuration was validated, switched, and refreshed.`,
        )
      } else {
        setSuccess(`${target.name} ${target.version} is active.`)
        setActionError({
          title: 'Recipe activated; dashboard refresh failed',
          message:
            'The Router switched successfully, but Models & Routing could not be reloaded. Reload this page before editing configuration.',
        })
      }
    } catch (reason) {
      if (controller.signal.aborted) return
      setActivationTarget(null)
      setActivationPlan(null)
      setActionError(presentRecipePackageError(reason, 'activate'))
    } finally {
      if (actionControllerRef.current === controller) {
        actionControllerRef.current = null
        if (mountedRef.current) setActiveOperation(null)
      }
    }
  }

  const handlePrepareActivation = async (target: RecipePackageSummary) => {
    if (operationPending || !canActivate || activationBlocked) return
    const controller = new AbortController()
    actionControllerRef.current?.abort()
    actionControllerRef.current = controller
    setActiveOperation(`preview:${target.recipe_digest}`)
    setActionError(null)
    try {
      const plan = await previewRecipePackageActivation(
        target.recipe_digest,
        target.warnings.length > 0,
        controller.signal,
      )
      if (controller.signal.aborted) return
      setActivationPlan(plan)
      setActivationTarget(target)
    } catch (reason) {
      if (!controller.signal.aborted)
        setActionError(presentRecipePackageError(reason, 'activate-preview'))
    } finally {
      if (actionControllerRef.current === controller) {
        actionControllerRef.current = null
        if (mountedRef.current) setActiveOperation(null)
      }
    }
  }

  const handleDeactivate = async () => {
    const target = deactivationTarget
    const plan = deactivationPlan
    if (!target || !plan || operationPending || !canActivate || activationState !== 'active') return

    const controller = new AbortController()
    actionControllerRef.current?.abort()
    actionControllerRef.current = controller
    setActiveOperation('deactivate')
    setActionError(null)
    setSuccess(null)
    try {
      await deactivateRecipePackage(controller.signal, {
        expected_plan_digest: plan.plan_digest,
        confirm_stack_recreation: plan.mode === 'stack_recreation',
      })
      if (controller.signal.aborted) return
      const configRefreshed = await onRecipeLifecycleChanged()
      if (controller.signal.aborted) return
      setDeactivationTarget(null)
      setDeactivationPlan(null)
      setPage(1)
      if (configRefreshed) {
        setSuccess(
          'The durable source runtime configuration captured before package management began was restored. Ordinary configuration editors are available again.',
        )
      } else {
        setSuccess('The source runtime configuration was restored.')
        setActionError({
          title: 'Source restored; dashboard refresh failed',
          message:
            'The Router restored the source configuration, but Models & Routing could not be reloaded. Reload this page before editing configuration.',
        })
      }
    } catch (reason) {
      if (controller.signal.aborted) return
      setDeactivationTarget(null)
      setDeactivationPlan(null)
      setActionError(presentRecipePackageError(reason, 'deactivate'))
    } finally {
      if (actionControllerRef.current === controller) {
        actionControllerRef.current = null
        if (mountedRef.current) setActiveOperation(null)
      }
    }
  }

  const handlePrepareDeactivation = async (target: RecipePackageSummary) => {
    if (operationPending || !canActivate || activationState !== 'active') return
    const controller = new AbortController()
    actionControllerRef.current?.abort()
    actionControllerRef.current = controller
    setActiveOperation('preview:deactivate')
    setActionError(null)
    try {
      const plan = await previewRecipePackageDeactivation(controller.signal)
      if (controller.signal.aborted) return
      setDeactivationPlan(plan)
      setDeactivationTarget(target)
    } catch (reason) {
      if (!controller.signal.aborted)
        setActionError(presentRecipePackageError(reason, 'deactivate-preview'))
    } finally {
      if (actionControllerRef.current === controller) {
        actionControllerRef.current = null
        if (mountedRef.current) setActiveOperation(null)
      }
    }
  }

  return (
    <section className={styles.panel} aria-labelledby="recipe-compatibility-title">
      <div className={styles.header}>
        <div>
          <span className={styles.eyebrow}>Backward compatibility</span>
          <h2 id="recipe-compatibility-title">Custom package lifecycle</h2>
          <p>
            Activate, repair, or leave custom packages that were already installed through the
            compatibility API. Official model distribution is the built-in catalog above.
          </p>
        </div>
        <span className={styles.total}>{inventory?.total ?? 0} installed</span>
      </div>

      {activationState === 'recovering' ? (
        <div className={styles.stateNotice} role="status">
          <strong>Activation recovery is in progress</strong>
          <span>
            Activation is temporarily disabled while the previous Router state is restored.
          </span>
        </div>
      ) : null}
      {activationBlocker === 'server_readonly' ? (
        <div className={styles.stateNotice} role="status">
          <strong>Custom package lifecycle is unavailable</strong>
          <span>This server has an explicit global read-only policy.</span>
        </div>
      ) : null}
      {activationBlocker === 'runtime_config_readonly' ? (
        <div className={styles.stateNotice} role="status">
          <strong>Runtime configuration is read-only</strong>
          <span>
            Activation, repair, and source restore require a writable runtime configuration mount.
          </span>
        </div>
      ) : null}
      {activationBlocker === 'permission_required' ? (
        <div className={styles.stateNotice} role="status">
          <strong>Recipe lifecycle permission required</strong>
          <span>
            Activating, restoring the source runtime, or repairing the Router requires config.deploy
            permission.
          </span>
        </div>
      ) : null}
      {activationState === 'inconsistent' ? (
        <div className={`${styles.stateNotice} ${styles.dangerNotice}`} role="alert">
          <strong>Router activation state is inconsistent</strong>
          <span>
            Routing and probes fail closed in this state.{' '}
            {canActivate
              ? 'Activate an installed package below to validate a known configuration and repair the Router.'
              : 'Restore runtime lifecycle capability before attempting repair.'}
          </span>
        </div>
      ) : null}
      {actionError ? (
        <div className={`${styles.stateNotice} ${styles.dangerNotice}`} role="alert">
          <strong>{actionError.title}</strong>
          <span>{actionError.message}</span>
          {actionError.code ? <code>{actionError.code}</code> : null}
        </div>
      ) : null}
      {success ? (
        <div className={`${styles.stateNotice} ${styles.successNotice}`} role="status">
          {success}
        </div>
      ) : null}

      {loadError ? (
        <div className={`${styles.stateNotice} ${styles.dangerNotice}`} role="alert">
          <strong>{loadError.title}</strong>
          <span>{loadError.message}</span>
          <button
            type="button"
            className={styles.secondaryButton}
            aria-label="Retry loading installed Recipe packages"
            onClick={() => setListRevision((current) => current + 1)}
          >
            Retry package inventory
          </button>
        </div>
      ) : null}

      <div className={styles.listToolbar}>
        <label>
          <span>Search installed packages</span>
          <input
            type="search"
            maxLength={256}
            value={query}
            placeholder="Recipe name, ID, version, or description"
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
        {query ? (
          <button
            type="button"
            className={styles.secondaryButton}
            aria-label="Clear installed Recipe package search"
            onClick={() => {
              setQuery('')
              setDebouncedQuery('')
              setPage(1)
            }}
          >
            Clear search
          </button>
        ) : null}
      </div>

      {loading && !inventory ? (
        <div className={styles.asyncState} role="status">
          Loading installed Recipe packages…
        </div>
      ) : null}

      {inventory ? (
        <div className={loading ? styles.refreshing : undefined} aria-busy={loading}>
          {inventory.items.length === 0 ? (
            <div className={styles.asyncState}>
              <strong>
                {debouncedQuery
                  ? 'No installed packages match this search'
                  : 'No Recipe packages installed'}
              </strong>
              <span>
                {debouncedQuery
                  ? 'Try a different name, ID, version, or digest.'
                  : 'Official models are available from the built-in catalogs above. This section only manages custom packages already recorded through the compatibility API.'}
              </span>
            </div>
          ) : (
            <ul className={styles.packageList} aria-label="Installed Recipe packages">
              {inventory.items.map((item) => {
                const digestID = item.recipe_digest.replace(/[^A-Za-z0-9_-]/g, '-')
                const titleID = `recipe-package-${digestID}`
                const isActivating = activeOperation === `activate:${item.recipe_digest}`
                const isActive = item.active && !repairMode
                return (
                  <li key={item.recipe_digest}>
                    <article className={styles.packageCard} aria-labelledby={titleID}>
                      <div className={styles.packageIdentity}>
                        <div className={styles.packageTitle}>
                          <div>
                            <h3 id={titleID}>{item.name}</h3>
                            <span>
                              {item.id} · {item.version}
                            </span>
                          </div>
                          {isActive ? <span className={styles.activeBadge}>Active</span> : null}
                        </div>
                        <p>{item.description}</p>
                        <span className={styles.installedAt}>
                          Installed {formatPackageInstalledAt(item.installed_at)}
                        </span>
                      </div>
                      <dl className={styles.packageFacts}>
                        <div>
                          <dt>Recipe digest</dt>
                          <dd>
                            <code title={item.recipe_digest}>
                              {shortDigest(item.recipe_digest)}
                            </code>
                          </dd>
                        </div>
                        <div>
                          <dt>Package record</dt>
                          <dd>
                            <span
                              className={
                                item.archive_verified ? styles.verifiedBadge : styles.recordedBadge
                              }
                            >
                              {item.archive_verified ? 'Integrity verified' : 'Digest recorded'}
                            </span>
                          </dd>
                        </div>
                      </dl>
                      <div className={styles.packageCounts} aria-label={`${item.name} inventory`}>
                        <span>{item.counts.unified_models} models</span>
                        <span>{item.counts.decisions} decisions</span>
                        <span>{item.counts.probes} probes</span>
                        {item.warnings.length ? (
                          <span className={styles.warningCount}>
                            {item.warnings.length} warning{item.warnings.length === 1 ? '' : 's'}
                          </span>
                        ) : null}
                      </div>
                      <div className={styles.packageActions}>
                        {isActive ? (
                          <>
                            <span>Serving now</span>
                            <button
                              type="button"
                              className={styles.secondaryButton}
                              disabled={operationPending || !canActivate}
                              title={!canActivate ? activationUnavailableTitle : undefined}
                              aria-label={`Restore source runtime from active Recipe package ${item.name} ${item.version} (${item.id})`}
                              onClick={() => void handlePrepareDeactivation(item)}
                            >
                              {activeOperation === 'deactivate' ? 'Restoring…' : 'Restore source'}
                            </button>
                          </>
                        ) : (
                          <button
                            type="button"
                            className={styles.primaryButton}
                            disabled={operationPending || !canActivate || activationBlocked}
                            title={!canActivate ? activationUnavailableTitle : undefined}
                            aria-label={`${repairMode ? 'Repair Router by activating' : 'Activate'} Recipe package ${item.name} ${item.version} (${item.id})`}
                            onClick={() => void handlePrepareActivation(item)}
                          >
                            {isActivating
                              ? 'Activating…'
                              : repairMode
                                ? 'Repair & activate'
                                : 'Activate'}
                          </button>
                        )}
                      </div>
                    </article>
                  </li>
                )
              })}
            </ul>
          )}

          {inventory.total > 0 ? (
            <div className={styles.pager}>
              <span>
                {inventory.total.toLocaleString()} packages · Page {inventory.page} of{' '}
                {Math.max(inventory.total_pages, 1)}
              </span>
              <div>
                <button
                  type="button"
                  className={styles.secondaryButton}
                  disabled={inventory.page <= 1 || loading}
                  aria-label="Previous installed Recipe packages page"
                  onClick={() => setPage((current) => current - 1)}
                >
                  Previous
                </button>
                <button
                  type="button"
                  className={styles.secondaryButton}
                  disabled={inventory.page >= inventory.total_pages || loading}
                  aria-label="Next installed Recipe packages page"
                  onClick={() => setPage((current) => current + 1)}
                >
                  Next
                </button>
              </div>
            </div>
          ) : null}
        </div>
      ) : null}

      <ConfirmDialog
        isOpen={Boolean(activationTarget)}
        title={
          activationTarget
            ? `${repairMode ? 'Repair with' : 'Activate'} ${activationTarget.name} ${activationTarget.version}?`
            : ''
        }
        eyebrow="Switch active Recipe"
        description={
          <>
            {repairMode
              ? 'Repair validates this installed Recipe and replaces the inconsistent Router configuration with its known package state. Routing services briefly restart during the switch.'
              : 'Activation validates this installed Recipe, switches the Router configuration, and briefly restarts routing services. If activation fails, the service attempts to restore the current Recipe.'}
          </>
        }
        details={
          activationTarget ? (
            <div className={styles.confirmContent}>
              <dl className={styles.confirmDetails}>
                <div>
                  <dt>Package</dt>
                  <dd>{activationTarget.id}</dd>
                </div>
                <div>
                  <dt>Recipe digest</dt>
                  <dd>
                    <code>{activationTarget.recipe_digest}</code>
                  </dd>
                </div>
                <div>
                  <dt>Runtime change</dt>
                  <dd>
                    {activationPlan?.mode === 'stack_recreation'
                      ? 'Stack recreation'
                      : 'Hot switch'}
                  </dd>
                </div>
              </dl>
              {activationPlan ? (
                <div className={styles.confirmWarnings}>
                  <strong>Activation plan</strong>
                  <ul>
                    {activationPlan.effects.map((effect) => (
                      <li key={effect}>{effect}</li>
                    ))}
                    {activationPlan.storage.add.length ? (
                      <li>Add managed storage: {activationPlan.storage.add.join(', ')}</li>
                    ) : null}
                    {activationPlan.storage.remove.length ? (
                      <li>Remove managed storage: {activationPlan.storage.remove.join(', ')}</li>
                    ) : null}
                    {activationPlan.storage.repair.length ? (
                      <li>Repair managed storage: {activationPlan.storage.repair.join(', ')}</li>
                    ) : null}
                  </ul>
                </div>
              ) : null}
              {activationTarget.warnings.length ? (
                <div className={styles.confirmWarnings}>
                  <strong>
                    Review and acknowledge {activationTarget.warnings.length} package warning
                    {activationTarget.warnings.length === 1 ? '' : 's'}
                  </strong>
                  <ul>
                    {activationTarget.warnings.map((warning, index) => {
                      const presentation = presentRecipePackageWarning(warning)
                      return (
                        <li key={`${presentation.code}-${presentation.path}-${index}`}>
                          <code>{presentation.code}</code>
                          {presentation.path ? <span>{presentation.path}</span> : null}
                          <p>{presentation.message}</p>
                        </li>
                      )
                    })}
                  </ul>
                </div>
              ) : null}
            </div>
          ) : null
        }
        confirmLabel={
          activationTarget?.warnings.length
            ? `Acknowledge warnings and ${repairMode ? 'repair' : 'activate'}`
            : activationTarget
              ? `${repairMode ? 'Repair with' : 'Activate'} ${activationTarget.name}`
              : 'Activate Recipe'
        }
        pending={activeOperation?.startsWith('activate:') ?? false}
        tone="warning"
        onCancel={() => {
          setActivationTarget(null)
          setActivationPlan(null)
        }}
        onConfirm={handleActivate}
      />
      <ConfirmDialog
        isOpen={Boolean(deactivationTarget)}
        title="Restore the source runtime configuration?"
        eyebrow="Leave package-managed runtime"
        description={
          <>
            This restores the durable source configuration captured before the first Recipe package
            activation and briefly restarts routing services. Installed packages are kept, and
            ordinary configuration editors become available again after a successful refresh.
          </>
        }
        details={
          deactivationTarget ? (
            <div className={styles.confirmContent}>
              <dl className={styles.confirmDetails}>
                <div>
                  <dt>Currently serving</dt>
                  <dd>{deactivationTarget.name}</dd>
                </div>
                <div>
                  <dt>Recipe digest</dt>
                  <dd>
                    <code>{deactivationTarget.recipe_digest}</code>
                  </dd>
                </div>
                <div>
                  <dt>Runtime change</dt>
                  <dd>
                    {deactivationPlan?.mode === 'stack_recreation'
                      ? 'Stack recreation'
                      : 'Hot switch'}
                  </dd>
                </div>
              </dl>
              {deactivationPlan ? (
                <div className={styles.confirmWarnings}>
                  <strong>Restore plan</strong>
                  <ul>
                    {deactivationPlan.effects.map((effect) => (
                      <li key={effect}>{effect}</li>
                    ))}
                    {deactivationPlan.storage.add.length ? (
                      <li>Add managed storage: {deactivationPlan.storage.add.join(', ')}</li>
                    ) : null}
                    {deactivationPlan.storage.remove.length ? (
                      <li>Remove managed storage: {deactivationPlan.storage.remove.join(', ')}</li>
                    ) : null}
                    {deactivationPlan.storage.repair.length ? (
                      <li>Repair managed storage: {deactivationPlan.storage.repair.join(', ')}</li>
                    ) : null}
                  </ul>
                </div>
              ) : null}
            </div>
          ) : null
        }
        confirmLabel="Restore source"
        pending={activeOperation === 'deactivate'}
        tone="warning"
        onCancel={() => {
          setDeactivationTarget(null)
          setDeactivationPlan(null)
        }}
        onConfirm={handleDeactivate}
      />
    </section>
  )
}
