import { useEffect, useMemo, useState } from 'react'
import DashboardSurfaceHero from '../components/DashboardSurfaceHero'
import RouterModelInventory from '../components/RouterModelInventory'
import { useModelResearch } from '../hooks/useModelResearch'
import type {
  ModelResearchCreateRequest,
  ModelResearchGoalTemplate,
  ModelResearchRecipeSummary,
} from '../types/modelResearch'
import {
  defaultCampaignName,
  formatImprovementPP,
  formatPercent,
  getCampaignStatusTone,
  getCampaignSubtitle,
  GOAL_TEMPLATE_OPTIONS,
} from './modelResearchPageSupport'
import ModelResearchTrendPanel from './ModelResearchTrendPanel'
import styles from './ModelResearchPage.module.css'

export default function ModelResearchPage() {
  const {
    recipesResponse,
    campaigns,
    selectedCampaign,
    selectedCampaignId,
    setSelectedCampaignId,
    targetsByGoal,
    loading,
    submitting,
    error,
    setError,
    createCampaign,
    stopSelectedCampaign,
  } = useModelResearch()

  const [goalTemplate, setGoalTemplate] = useState<ModelResearchGoalTemplate>('improve_accuracy')
  const [target, setTarget] = useState('')
  const [name, setName] = useState('')
  const [maxTrials, setMaxTrials] = useState(2)
  const [successThresholdPP, setSuccessThresholdPP] = useState(0.5)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [apiBaseOverride, setAPIBaseOverride] = useState('')
  const [requestModelOverride, setRequestModelOverride] = useState('')
  const [datasetOverride, setDatasetOverride] = useState('')
  const [hyperparameterHints, setHyperparameterHints] = useState('')
  const [allowCPUDryRun, setAllowCPUDryRun] = useState(false)

  const targetOptions = targetsByGoal[goalTemplate]
  const selectedRecipe = useMemo<ModelResearchRecipeSummary | null>(() => {
    return targetOptions.find((recipe) => recipe.key === target) ?? targetOptions[0] ?? null
  }, [target, targetOptions])

  useEffect(() => {
    if (!selectedRecipe) {
      setTarget('')
      return
    }
    if (selectedRecipe.key !== target) {
      setTarget(selectedRecipe.key)
    }
  }, [selectedRecipe, target])

  useEffect(() => {
    if (!selectedRecipe) return
    setSuccessThresholdPP(selectedRecipe.default_success_threshold_pp)
    setName((current) => current || defaultCampaignName(goalTemplate, selectedRecipe.key))
  }, [goalTemplate, selectedRecipe])

  const runtimeModels = selectedCampaign?.runtime_models ?? recipesResponse?.runtime_models ?? null
  const selectedGoalOption = GOAL_TEMPLATE_OPTIONS.find((option) => option.value === goalTemplate) ?? GOAL_TEMPLATE_OPTIONS[0]
  const runtimeModelCount = runtimeModels?.summary?.loaded_models ?? runtimeModels?.models?.length ?? 0

  const handleStartCampaign = async () => {
    if (!selectedRecipe) return

    let parsedHints: Record<string, unknown> | undefined
    if (hyperparameterHints.trim()) {
      try {
        parsedHints = JSON.parse(hyperparameterHints) as Record<string, unknown>
      } catch {
        setError('Advanced hyperparameter hints must be valid JSON')
        return
      }
    }

    const payload: ModelResearchCreateRequest = {
      name: name.trim() || defaultCampaignName(goalTemplate, selectedRecipe.key),
      goal_template: goalTemplate,
      target: selectedRecipe.key,
      budget: { max_trials: maxTrials },
      success_threshold_pp: successThresholdPP,
      overrides: {
        api_base_override: apiBaseOverride.trim() || undefined,
        request_model_override: requestModelOverride.trim() || undefined,
        dataset_override: datasetOverride.trim() || undefined,
        hyperparameter_hints: parsedHints,
        allow_cpu_dry_run: allowCPUDryRun,
      },
    }

    await createCampaign(payload)
  }

  return (
    <div className={styles.page}>
      <DashboardSurfaceHero
        eyebrow="Operations"
        title="Auto Research"
        description="Run AMD-first classifier research loops against the current router, keep MoM as the default request model, and confine any API/model overrides to the current campaign."
        meta={[
          { label: 'Current surface', value: selectedGoalOption.label },
          { label: 'Tracked campaigns', value: `${campaigns.length}` },
          { label: 'Runtime baselines', value: `${runtimeModelCount} models` },
        ]}
        panelEyebrow="Router defaults"
        panelTitle="MoM-aligned classifier loops"
        panelDescription="The backend reuses the connected router target for runtime discovery and eval. Advanced overrides stay campaign-local and never rewrite dashboard-wide settings."
        pills={GOAL_TEMPLATE_OPTIONS.map((option) => ({
          label: option.label,
          active: option.value === goalTemplate,
          onClick: () => setGoalTemplate(option.value),
        }))}
        panelFooter={
          <div className={styles.heroFacts}>
            <div className={styles.heroFact}>
              <span>Default platform</span>
              <strong>{recipesResponse?.default_platform || 'amd'}</strong>
            </div>
            <div className={styles.heroFact}>
              <span>Request model</span>
              <strong>{recipesResponse?.default_request_model || 'MoM'}</strong>
            </div>
            <div className={`${styles.heroFact} ${styles.heroFactWide}`}>
              <span>API base</span>
              <strong>{recipesResponse?.default_api_base || 'Router managed by dashboard settings'}</strong>
            </div>
          </div>
        }
      />

      {error && (
        <div className={styles.errorBanner}>
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)} aria-label="Dismiss error">
            ×
          </button>
        </div>
      )}

      <div className={styles.body}>
        <div className={styles.layout}>
          <aside className={styles.sidebar}>
            <div className={styles.sidebarHeader}>
              <div>
                <h2>Campaigns</h2>
                <p>Recent research runs and their best observed lift.</p>
              </div>
              <span>{campaigns.length}</span>
            </div>
            <div className={styles.sidebarList}>
              {campaigns.length === 0 && !loading ? (
                <div className={styles.emptyState}>No campaigns yet.</div>
              ) : (
                campaigns.map((campaign) => (
                  <button
                    key={campaign.id}
                    type="button"
                    className={`${styles.campaignButton} ${selectedCampaignId === campaign.id ? styles.campaignButtonActive : ''}`}
                    onClick={() => setSelectedCampaignId(campaign.id)}
                  >
                    <div className={styles.campaignButtonRow}>
                      <span>{campaign.name}</span>
                      <span
                        className={`${styles.statusBadge} ${styles[`status${getCampaignStatusTone(campaign.status)}`]}`}
                      >
                        {campaign.status}
                      </span>
                    </div>
                    <span className={styles.campaignHint}>{getCampaignSubtitle(campaign)}</span>
                  </button>
                ))
              )}
            </div>
          </aside>

          <main className={styles.main}>
            <section className={styles.panel}>
              <div className={styles.panelHeader}>
                <div>
                  <h2>Create Campaign</h2>
                  <p>{selectedGoalOption.description}</p>
                </div>
                <button
                  type="button"
                  className={styles.secondaryButton}
                  onClick={() => setShowAdvanced((value) => !value)}
                >
                  {showAdvanced ? 'Hide advanced controls' : 'Show advanced controls'}
                </button>
              </div>

              <div className={styles.formGrid}>
                <label className={styles.field}>
                  <span>Campaign name</span>
                  <input
                    value={name}
                    onChange={(event) => setName(event.target.value)}
                    placeholder="accuracy-feedback-2026-03-19"
                  />
                </label>

                <label className={styles.field}>
                  <span>Target classifier / signal</span>
                  <select value={target} onChange={(event) => setTarget(event.target.value)}>
                    {targetOptions.map((recipe) => (
                      <option key={recipe.key} value={recipe.key}>
                        {recipe.label}
                      </option>
                    ))}
                  </select>
                </label>

                <label className={styles.field}>
                  <span>Budget (max trials)</span>
                  <input
                    type="number"
                    min={1}
                    max={3}
                    value={maxTrials}
                    onChange={(event) => setMaxTrials(Number(event.target.value) || 1)}
                  />
                </label>

                <label className={styles.field}>
                  <span>Success threshold (pp)</span>
                  <input
                    type="number"
                    min={0.1}
                    step={0.1}
                    value={successThresholdPP}
                    onChange={(event) => setSuccessThresholdPP(Number(event.target.value) || 0.5)}
                  />
                </label>
              </div>

              {selectedRecipe && (
                <div className={styles.recipeCard}>
                  <div className={styles.recipeCardHeader}>
                    <h3>{selectedRecipe.label}</h3>
                    <span className={styles.inlinePill}>{selectedRecipe.primary_metric}</span>
                  </div>
                  <p>{selectedRecipe.dataset_hint}</p>
                  <div className={styles.recipeMeta}>
                    <span>Default dataset: {selectedRecipe.default_dataset}</span>
                    <span>
                      Baseline:{' '}
                      {selectedRecipe.baseline.model_id ||
                        selectedRecipe.baseline.model_path ||
                        'MoM runtime'}
                    </span>
                  </div>
                </div>
              )}

              {showAdvanced && (
                <div className={styles.advancedPanel}>
                  <div className={styles.formGrid}>
                    <label className={styles.field}>
                      <span>API base override</span>
                      <input
                        value={apiBaseOverride}
                        onChange={(event) => setAPIBaseOverride(event.target.value)}
                        placeholder={recipesResponse?.default_api_base || 'http://localhost:8080'}
                      />
                    </label>

                    <label className={styles.field}>
                      <span>Request model override</span>
                      <input
                        value={requestModelOverride}
                        onChange={(event) => setRequestModelOverride(event.target.value)}
                        placeholder={recipesResponse?.default_request_model || 'MoM'}
                      />
                    </label>

                    <label className={styles.field}>
                      <span>Dataset override</span>
                      <input
                        value={datasetOverride}
                        onChange={(event) => setDatasetOverride(event.target.value)}
                        placeholder={selectedRecipe?.default_dataset || ''}
                      />
                    </label>

                    <label className={`${styles.field} ${styles.checkboxField}`}>
                      <input
                        type="checkbox"
                        checked={allowCPUDryRun}
                        onChange={(event) => setAllowCPUDryRun(event.target.checked)}
                      />
                      <span>Allow CPU dry run when AMD is unavailable</span>
                    </label>
                  </div>

                  <label className={styles.field}>
                    <span>Partial hyperparameter hints (JSON)</span>
                    <textarea
                      value={hyperparameterHints}
                      onChange={(event) => setHyperparameterHints(event.target.value)}
                      placeholder='{"epochs": 6, "learning_rate": 0.00002}'
                      rows={6}
                    />
                  </label>
                </div>
              )}

              <div className={styles.panelActions}>
                <button
                  type="button"
                  className={styles.primaryButton}
                  onClick={handleStartCampaign}
                  disabled={submitting || !selectedRecipe}
                >
                  {submitting ? 'Starting…' : 'Start campaign'}
                </button>
              </div>
            </section>

            <section className={styles.panel}>
              <div className={styles.panelHeader}>
                <div>
                  <h2>Runtime baseline</h2>
                  <p>
                    Read-only inventory discovered from the current router. These models seed the
                    baseline shown in each recipe.
                  </p>
                </div>
              </div>
              <RouterModelInventory
                modelsInfo={runtimeModels}
                mode="preview"
                previewLimit={4}
                emptyMessage="No runtime models discovered from /info/models."
              />
            </section>

            {selectedCampaign && (
              <section className={styles.panel}>
                <div className={styles.panelHeader}>
                  <div>
                    <h2>{selectedCampaign.name}</h2>
                    <p>{selectedCampaign.recipe.label}</p>
                  </div>
                  {['pending', 'running'].includes(selectedCampaign.status) && (
                    <button
                      type="button"
                      className={styles.stopButton}
                      onClick={() => void stopSelectedCampaign()}
                    >
                      Stop campaign
                    </button>
                  )}
                </div>

                <div className={styles.statsGrid}>
                  <StatCard
                    label="Status"
                    value={selectedCampaign.status}
                    tone={getCampaignStatusTone(selectedCampaign.status)}
                  />
                  <StatCard label="Platform" value={selectedCampaign.platform} />
                  <StatCard
                    label="Baseline accuracy"
                    value={formatPercent(selectedCampaign.baseline_eval?.accuracy)}
                  />
                  <StatCard
                    label="Best improvement"
                    value={formatImprovementPP(selectedCampaign.best_trial?.eval?.improvement_pp)}
                  />
                  <StatCard label="API base" value={selectedCampaign.api_base} compact />
                  <StatCard label="Request model" value={selectedCampaign.request_model} />
                </div>

                <ModelResearchTrendPanel campaign={selectedCampaign} />

                <div className={styles.detailGrid}>
                  <div className={styles.detailCard}>
                    <h3>Baseline</h3>
                    <p>{selectedCampaign.baseline.description || 'Current runtime baseline'}</p>
                    <div className={styles.detailList}>
                      <span>Source: {selectedCampaign.baseline.source}</span>
                      <span>
                        Model:{' '}
                        {selectedCampaign.baseline.model_id ||
                          selectedCampaign.baseline.model_path ||
                          'N/A'}
                      </span>
                      <span>Dataset: {selectedCampaign.recipe.default_dataset}</span>
                      {selectedCampaign.runtime_baseline && (
                        <span>
                          Runtime signal eval:{' '}
                          {formatPercent(selectedCampaign.runtime_baseline.accuracy)}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className={styles.detailCard}>
                    <h3>Best candidate</h3>
                    {selectedCampaign.best_trial?.eval ? (
                      <div className={styles.detailList}>
                        <span>Trial: {selectedCampaign.best_trial.name}</span>
                        <span>
                          Accuracy: {formatPercent(selectedCampaign.best_trial.eval.accuracy)}
                        </span>
                        <span>F1: {formatPercent(selectedCampaign.best_trial.eval.f1)}</span>
                        <span>
                          Improvement:{' '}
                          {formatImprovementPP(
                            selectedCampaign.best_trial.eval.improvement_pp
                          )}
                        </span>
                        <span>Model path: {selectedCampaign.best_trial.model_path || 'N/A'}</span>
                        <span>
                          Config fragment: {selectedCampaign.config_fragment_path || 'Pending'}
                        </span>
                      </div>
                    ) : (
                      <p>No successful trial yet.</p>
                    )}
                  </div>
                </div>

                <div className={styles.tableSection}>
                  <h3>Trials</h3>
                  <table className={styles.table}>
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>Status</th>
                        <th>Accuracy</th>
                        <th>Improvement</th>
                        <th>Model</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(selectedCampaign.trials ?? []).map((trial) => (
                        <tr key={trial.name}>
                          <td>{trial.name}</td>
                          <td>{trial.status}</td>
                          <td>{formatPercent(trial.eval?.accuracy)}</td>
                          <td>{formatImprovementPP(trial.eval?.improvement_pp)}</td>
                          <td className={styles.monoCell}>{trial.model_path || 'Pending'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className={styles.logSection}>
                  <h3>Live log</h3>
                  <div className={styles.logList}>
                    {(selectedCampaign.events ?? []).map((event, index) => (
                      <div key={`${event.timestamp}-${index}`} className={styles.logItem}>
                        <span className={styles.logTime}>
                          {new Date(event.timestamp).toLocaleTimeString()}
                        </span>
                        <span className={styles.logLevel}>{event.level || event.kind}</span>
                        <span className={styles.logMessage}>{event.message}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </section>
            )}
          </main>
        </div>
      </div>
    </div>
  )
}

function StatCard({
  label,
  value,
  tone = 'muted',
  compact = false,
}: {
  label: string
  value: string
  tone?: 'good' | 'warn' | 'bad' | 'muted'
  compact?: boolean
}) {
  return (
    <div className={`${styles.statCard} ${styles[`status${tone}`]} ${compact ? styles.statCardCompact : ''}`}>
      <span className={styles.statLabel}>{label}</span>
      <span className={styles.statValue}>{value}</span>
    </div>
  )
}
