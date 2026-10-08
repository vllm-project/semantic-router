import { useState } from 'react'
import { Link } from 'react-router-dom'
import qwenLogo from '@lobehub/icons-static-svg/icons/qwen-color.svg'
import { DECISION_MODEL_OPTIONS } from './decisionModelSupport'
import {
  DECISION_PROVIDERS,
  DECISION_RUNTIME_CAPABILITIES,
  DECISION_RUNTIME_CATALOG,
  runtimeBackboneLabel,
  runtimeFamilyLabel,
  type DecisionRuntimeCatalogEntry,
} from './decisionRuntimeCatalog'
import {
  consumerLabel,
  configuredDecisionRuntimes,
  decisionRuntimeConsumers,
} from './decisionRuntimeDeployment'
import type { useDecisionModelManagement } from './useDecisionModelManagement'
import DecisionRuntimeDeployDialog from './DecisionRuntimeDeployDialog'
import SystemOneSelect from './SystemOneSelect'
import styles from './DecisionModelCatalog.module.css'
import pageStyles from './DecisionModelPage.module.css'

interface Props {
  model: ReturnType<typeof useDecisionModelManagement>
  writable: boolean
  engineOnly: boolean
}

const families = [
  {
    value: 'all',
    label: 'All families',
    count: DECISION_MODEL_OPTIONS.length + DECISION_RUNTIME_CATALOG.length,
  },
  {
    value: 'vela2',
    label: 'Vela 2.0',
    count: DECISION_MODEL_OPTIONS.filter((entry) => entry.name !== 'Vela-1.0').length,
  },
  { value: 'vela1', label: 'Vela 1.0', count: 1 },
  ...[...new Set(DECISION_RUNTIME_CATALOG.map((entry) => entry.family))].map((family) => ({
    value: family,
    label: runtimeFamilyLabel(DECISION_RUNTIME_CATALOG.find((entry) => entry.family === family)!),
    count: DECISION_RUNTIME_CATALOG.filter((entry) => entry.family === family).length,
  })),
]

export default function DecisionModelCatalog({ model, writable, engineOnly }: Props) {
  const [search, setSearch] = useState('')
  const [family, setFamily] = useState('all')
  const [capability, setCapability] = useState('all')
  const [provider, setProvider] = useState('all')
  const [dialog, setDialog] = useState<{
    entry: DecisionRuntimeCatalogEntry
    existingName?: string
  } | null>(null)
  const query = search.trim().toLowerCase()
  const matches = (...values: string[]) => values.join(' ').toLowerCase().includes(query)
  const builtins = DECISION_MODEL_OPTIONS.filter((option) => {
    const isVela1 = option.name === 'Vela-1.0'
    return (
      (family === 'all' || family === (isVela1 ? 'vela1' : 'vela2')) &&
      (provider === 'all' || provider === 'vllm-sr') &&
      (capability === 'all' || !isVela1) &&
      matches(
        option.name,
        option.label,
        option.hardware,
        'vllm-sr',
        DECISION_PROVIDERS['vllm-sr'].name,
      )
    )
  })
  const runtimes = DECISION_RUNTIME_CATALOG.filter(
    (entry) =>
      (family === 'all' || family === entry.family) &&
      (provider === 'all' || provider === entry.provider) &&
      (capability === 'all' || DECISION_RUNTIME_CAPABILITIES.some((kind) => kind === capability)) &&
      matches(
        entry.name,
        entry.id,
        runtimeFamilyLabel(entry),
        runtimeBackboneLabel(entry),
        DECISION_PROVIDERS[entry.provider]?.name ?? entry.provider,
      ),
  )
  const declarations = configuredDecisionRuntimes(model.config, model.inventory)
  const consumers = decisionRuntimeConsumers(model.config)
  const disabled = !writable || model.deploying || !model.global
  return (
    <div className={styles.catalog}>
      <div className={styles.catalogHeader}>
        <div>
          <span className={styles.eyebrow}>Model library</span>
          <h2 id="decision-model-choose-title">Choose a decision model</h2>
          <p className={styles.help}>
            Explore router intelligence and custom decision runtimes by provider, family and
            capability.
          </p>
        </div>
        <span className={styles.catalogCount}>
          {DECISION_MODEL_OPTIONS.length + DECISION_RUNTIME_CATALOG.length} models
        </span>
      </div>
      <div className={styles.filters}>
        <label className={styles.search}>
          <span className={pageStyles.srOnly}>Search decision models</span>
          <svg
            viewBox="0 0 24 24"
            width="18"
            height="18"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.7"
            aria-hidden="true"
          >
            <circle cx="10.5" cy="10.5" r="6.5" />
            <path d="m16 16 4 4" />
          </svg>
          <input
            placeholder="Search models, providers or architectures…"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </label>
        <SystemOneSelect
          label="Model provider"
          value={provider}
          onChange={setProvider}
          options={[
            { value: 'all', label: 'All providers' },
            ...Object.entries(DECISION_PROVIDERS).map(([value, item]) => ({
              value,
              label: item.name,
            })),
          ]}
        />
        <SystemOneSelect
          label="Question capability"
          value={capability}
          onChange={setCapability}
          options={[
            { value: 'all', label: 'All capabilities' },
            ...['choice', 'score', 'noul', 'span', 'set'].map((value) => ({ value, label: value })),
          ]}
        />
      </div>
      <div className={styles.familyTabs} aria-label="Model families">
        {families.map((item) => (
          <button
            key={item.value}
            type="button"
            aria-pressed={family === item.value}
            onClick={() => setFamily(item.value)}
          >
            {item.label}
            <span>{item.count}</span>
          </button>
        ))}
      </div>
      {engineOnly && (
        <p className={pageStyles.notice}>
          This deployment is a standalone engine. Connect this Dashboard to a router to manage model
          configuration and bindings.
        </p>
      )}
      {builtins.length > 0 && (
        <section className={styles.familySection} aria-labelledby="router-intelligence-models">
          <header className={styles.sectionHeader}>
            <div>
              <h3 id="router-intelligence-models">Router intelligence</h3>
              <p>
                Vela powers built-in signals. Vela 2.0 also answers custom questions that follow the
                router default.
              </p>
            </div>
            <span className={styles.roleBadge}>Router default</span>
          </header>
          <fieldset className={styles.cards} disabled={disabled}>
            <legend className={pageStyles.srOnly}>Decision model selection</legend>
            {builtins.map((option) => (
              <label
                key={option.name}
                className={`${styles.card} ${model.selectedModel === option.name ? styles.selected : ''}`}
              >
                <div className={styles.cardTop}>
                  <img
                    className={styles.logo}
                    src={DECISION_PROVIDERS['vllm-sr'].logo}
                    alt="vLLM Semantic Router"
                  />
                  <input
                    type="radio"
                    name="decision-model"
                    value={option.name}
                    checked={model.selectedModel === option.name}
                    onChange={() => model.selectModel(option.name)}
                  />
                </div>
                <span className={styles.providerName}>vllm-sr</span>
                <strong className={styles.modelName}>{option.label}</strong>
                <span className={styles.description}>{option.summary}</span>
                <div className={styles.capabilities}>
                  {(option.name === 'Vela-1.0'
                    ? ['Built-in specialists']
                    : ['choice', 'score', 'noul', 'span', 'set']
                  ).map((kind) => (
                    <span key={kind}>{kind}</span>
                  ))}
                </div>
                <div className={styles.hardware}>{option.hardware}</div>
                <span className={styles.cardState}>
                  {model.savedModel === option.name
                    ? 'Saved configuration'
                    : model.selectedModel === option.name
                      ? 'Selected · not saved'
                      : option.name === 'Vela-2.0-0.3B'
                        ? 'Default for new routers'
                        : 'Available'}
                </span>
              </label>
            ))}
          </fieldset>
          <div className={styles.selectionBar}>
            <span>
              {writable ? (
                <>
                  Router default: <strong>{model.selectedModel}</strong>
                  <small>Existing explicit bindings stay unchanged.</small>
                </>
              ) : (
                'This session can inspect models. Configuration write access is required to deploy.'
              )}
            </span>
            <button
              type="button"
              className={styles.primary}
              disabled={disabled || !model.selectedModel}
              onClick={() => void model.deploy()}
            >
              {model.deploying ? 'Deploying…' : 'Deploy selected model'}
            </button>
          </div>
        </section>
      )}
      {runtimes.length > 0 && (
        <section className={styles.familySection} aria-labelledby="custom-decision-models">
          <header className={styles.sectionHeader}>
            <div>
              <h3 id="custom-decision-models">Custom decision runtimes</h3>
              <p>
                Deploy with an explicit question or decision-selector binding. Your Vela router
                default stays in place.
              </p>
            </div>
            <span className={styles.roleBadge}>Per-consumer deployment</span>
          </header>
          <div className={styles.cards}>
            {runtimes.map((entry) => {
              const publishedBy = DECISION_PROVIDERS[entry.provider]
              const configured = declarations.filter(
                ([, deployment]) => deployment.artifact === entry.id,
              )
              return (
                <article key={entry.id} className={styles.card} aria-label={entry.name}>
                  <div className={styles.cardTop}>
                    {publishedBy ? (
                      <img className={styles.logo} src={publishedBy.logo} alt={publishedBy.name} />
                    ) : (
                      <span className={styles.logoFallback}>{entry.provider.slice(0, 2)}</span>
                    )}
                    <span className={styles.versionBadge}>{runtimeFamilyLabel(entry)}</span>
                  </div>
                  <span className={styles.providerName}>{entry.provider}</span>
                  <h4 className={styles.modelName}>
                    {entry.name.replace(/^Decision-[12]\.0-/, '').replace(/-/g, ' ')}
                  </h4>
                  <p className={styles.description}>
                    {entry.backbone === 'modernbert'
                      ? 'Encoder decision model for custom classification and scoring.'
                      : 'Decoder decision model for custom classification, scoring and model selection.'}
                  </p>
                  <div className={styles.capabilities}>
                    {DECISION_RUNTIME_CAPABILITIES.map((kind) => (
                      <span key={kind}>{kind}</span>
                    ))}
                  </div>
                  <div className={styles.modelFacts}>
                    <span>
                      {entry.backbone.startsWith('qwen') && <img src={qwenLogo} alt="" />}
                      {runtimeBackboneLabel(entry)}
                    </span>
                    <span>≥ {entry.minMemoryGiB} GiB</span>
                  </div>
                  <div className={styles.cardFooter}>
                    <span className={styles.cardState}>
                      {configured.length ? `${configured.length} configured` : 'Available'}
                    </span>
                    <button
                      type="button"
                      onClick={() => setDialog({ entry })}
                      disabled={model.deploying || !model.config}
                    >
                      {writable ? 'Configure' : 'View model'}
                    </button>
                  </div>
                </article>
              )
            })}
          </div>
        </section>
      )}
      {!builtins.length && !runtimes.length && (
        <div className={styles.empty}>
          <strong>No models match these filters</strong>
          <p>Try another name, family or question capability.</p>
          <button
            type="button"
            onClick={() => {
              setSearch('')
              setFamily('all')
              setProvider('all')
              setCapability('all')
            }}
          >
            Clear filters
          </button>
        </div>
      )}
      {declarations.length > 0 && (
        <section className={styles.familySection} aria-labelledby="configured-decision-runtimes">
          <header className={styles.sectionHeader}>
            <div>
              <h3 id="configured-decision-runtimes">Configured runtimes</h3>
              <p>
                Decision models and explicit question or selector bindings. Live inventory reports
                readiness.
              </p>
            </div>
            <Link to="/decision-model/monitoring">Monitor all →</Link>
          </header>
          <div className={styles.deployments}>
            {declarations.map(([name, deployment]) => {
              const entry = DECISION_RUNTIME_CATALOG.find((item) => item.id === deployment.artifact)
              const runtime = model.inventory?.deployments.find((item) => item.name === name)
              const uses = consumers.filter((consumer) => consumer.deployment === name)
              const state = runtime
                ? runtime.ready && runtime.state === 'ready'
                  ? 'Ready'
                  : runtime.state || 'Not ready'
                : uses.length
                  ? 'Not reported'
                  : 'Saved · no decision binding'
              return (
                <div key={name} className={styles.deployment}>
                  <div>
                    <strong>{name}</strong>
                    <p>{deployment.artifact || deployment.endpoint || 'External runtime'}</p>
                    <small>
                      {uses.length
                        ? uses.map(consumerLabel).join(' · ')
                        : 'No decision question or selector is bound.'}
                    </small>
                    {runtime?.reason && <p>{runtime.reason}</p>}
                  </div>
                  <span className={`${styles.state} ${state === 'Ready' ? styles.ready : ''}`}>
                    {state}
                  </span>
                  {entry && !deployment.endpoint ? (
                    <button
                      type="button"
                      onClick={() => setDialog({ entry, existingName: name })}
                      disabled={model.deploying}
                    >
                      Manage
                    </button>
                  ) : (
                    <Link to="/config/global-config#global-section-system_models">
                      Advanced configuration →
                    </Link>
                  )}
                </div>
              )
            })}
          </div>
        </section>
      )}
      {dialog && (
        <DecisionRuntimeDeployDialog
          key={`${dialog.entry.id}:${dialog.existingName ?? ''}`}
          {...dialog}
          config={model.config}
          writable={writable}
          onDeploy={model.deployRuntime}
          onClose={() => setDialog(null)}
        />
      )}
    </div>
  )
}
