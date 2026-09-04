import ProductIcon from '../components/ProductIcon'
import ConfigPageConnectModelAdvanced from './ConfigPageConnectModelAdvanced'
import type { ConnectModelsDialogController } from './configPageConnectModelsDialogController'
import { requestedConnectedModelName } from './configPageConnectModelSupport'
import ModelProviderLogo from './ModelProviderLogo'
import type { ModelProviderPreset } from './modelProviderCatalog'
import styles from './ConfigPageConnectModelsDialog.module.css'

const providerCategories = ['Start here', 'Model APIs', 'Private runtimes'] as const

interface Props {
  controller: ConnectModelsDialogController
  reasoningFamilies: string[]
  onClose: () => void
  onManualSetup: () => void
}

export default function ConfigPageConnectModelsDialogView({
  controller,
  reasoningFamilies,
  onClose,
  onManualSetup,
}: Props) {
  return (
    <div
      className={styles.backdrop}
      onMouseDown={(event) => event.target === event.currentTarget && !controller.busy && onClose()}
    >
      <div
        ref={controller.dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={controller.titleId}
        tabIndex={-1}
      >
        <DialogHeader controller={controller} onClose={onClose} />
        {controller.error ? (
          <div className={styles.error} role="alert">
            {controller.error}
          </div>
        ) : null}
        {controller.stage === 'provider' ? (
          <ProviderStage controller={controller} />
        ) : controller.provider ? (
          <ModelsStage controller={controller} reasoningFamilies={reasoningFamilies} />
        ) : null}
        <DialogFooter controller={controller} onClose={onClose} onManualSetup={onManualSetup} />
      </div>
    </div>
  )
}

function DialogHeader({ controller, onClose }: Pick<Props, 'controller' | 'onClose'>) {
  const choosingModels = controller.stage === 'models' && controller.provider
  return (
    <header className={styles.header}>
      <div className={styles.titleGroup}>
        {choosingModels ? (
          <button
            type="button"
            className={styles.backButton}
            onClick={() => controller.setStage('provider')}
            disabled={controller.busy}
            aria-label="Choose another provider"
          >
            <ProductIcon name="arrow-left" aria-hidden="true" />
          </button>
        ) : null}
        {choosingModels ? (
          <ModelProviderLogo provider={controller.provider!} size="medium" />
        ) : null}
        <div>
          <h2 id={controller.titleId}>
            {controller.stage === 'provider'
              ? 'Add models'
              : (controller.provider?.name ?? 'Connect provider')}
          </h2>
          <p>
            {controller.stage === 'provider'
              ? 'Choose where your models run.'
              : 'Connect once, then import one or many models.'}
          </p>
        </div>
      </div>
      <button
        type="button"
        className={styles.iconButton}
        onClick={onClose}
        disabled={controller.busy}
        aria-label="Close"
      >
        <ProductIcon name="close" />
      </button>
    </header>
  )
}

function ProviderStage({ controller }: Pick<Props, 'controller'>) {
  return (
    <div className={styles.body}>
      <div className={styles.searchShell}>
        <ProductIcon name="search" aria-hidden="true" />
        <input
          className={styles.search}
          type="search"
          value={controller.search}
          onChange={(event) => controller.setSearch(event.target.value)}
          placeholder="Search providers"
          autoFocus
          data-dialog-initial-focus
        />
        <small>{controller.visibleProviders.length} providers</small>
      </div>
      {providerCategories.map((category) => (
        <ProviderCategory
          key={category}
          category={category}
          providers={controller.visibleProviders.filter((item) => item.category === category)}
          onChoose={controller.chooseProvider}
        />
      ))}
    </div>
  )
}

function ProviderCategory({
  category,
  providers,
  onChoose,
}: {
  category: string
  providers: ModelProviderPreset[]
  onChoose: (provider: ModelProviderPreset) => void
}) {
  if (providers.length === 0) return null
  return (
    <section className={styles.providerSection}>
      <h3>{category}</h3>
      <div className={styles.providerGrid}>
        {providers.map((provider) => (
          <button
            key={provider.id}
            type="button"
            className={styles.providerCard}
            onClick={() => onChoose(provider)}
          >
            <ModelProviderLogo provider={provider} size="medium" />
            <span>
              <strong>{provider.name}</strong>
              <small>{provider.description}</small>
            </span>
          </button>
        ))}
      </div>
    </section>
  )
}

function ModelsStage({
  controller,
  reasoningFamilies,
}: Pick<Props, 'controller' | 'reasoningFamilies'>) {
  return (
    <div className={styles.body}>
      <ConnectionPanel controller={controller} />
      <ManualModelRow controller={controller} />
      {controller.models.length > 0 ? <ModelPicker controller={controller} /> : null}
      <ConfigPageConnectModelAdvanced
        value={controller.advanced}
        reasoningFamilies={reasoningFamilies}
        onChange={controller.setAdvanced}
      />
    </div>
  )
}

function ConnectionPanel({ controller }: Pick<Props, 'controller'>) {
  const provider = controller.provider!
  return (
    <section className={styles.connectionPanel}>
      <div className={styles.connectionHeading}>
        <div>
          <strong>Connection</strong>
          <span>
            {provider.supportsModelDiscovery
              ? 'Credentials stay private and are only used for this provider.'
              : 'This provider does not expose catalog-backed discovery; enter a model ID below.'}
          </span>
        </div>
        {provider.supportsModelDiscovery ? (
          <button
            type="button"
            className={styles.discoverButton}
            onClick={() => void controller.discover()}
            disabled={controller.busy || !controller.baseUrl.trim()}
          >
            <ProductIcon
              name={controller.models.length > 0 ? 'refresh' : 'search'}
              aria-hidden="true"
            />
            {controller.discovering
              ? 'Connecting…'
              : controller.models.length > 0
                ? 'Refresh'
                : 'List models'}
          </button>
        ) : null}
      </div>
      <ConnectionFields controller={controller} provider={provider} />
    </section>
  )
}

function ConnectionFields({
  controller,
  provider,
}: Pick<Props, 'controller'> & { provider: ModelProviderPreset }) {
  return (
    <div className={styles.connectionGrid}>
      {!provider.baseUrl ? (
        <label className={styles.field}>
          <span>Base URL</span>
          <input
            type="url"
            value={controller.baseUrl}
            onChange={(event) => controller.setBaseUrl(event.target.value)}
            placeholder="https://api.example.com/v1"
            autoFocus
          />
        </label>
      ) : (
        <div className={`${styles.field} ${styles.fixedEndpoint}`}>
          <span>API endpoint</span>
          <div>
            <ProductIcon name="link" aria-hidden="true" />
            <code>{provider.baseUrl}</code>
          </div>
        </div>
      )}
      {provider.authStrategy !== 'none' ? (
        <label className={styles.field}>
          <span>
            API key <small>{provider.baseUrl ? 'Required' : 'Optional'}</small>
          </span>
          <input
            type="password"
            value={controller.apiKey}
            onChange={(event) => controller.setAPIKey(event.target.value)}
            placeholder="Paste your key"
            autoComplete="new-password"
            autoFocus={Boolean(provider.baseUrl)}
          />
        </label>
      ) : (
        <div className={`${styles.field} ${styles.noCredential}`}>
          <span>Authentication</span>
          <div>
            <ProductIcon name="check" aria-hidden="true" />
            No API key required
          </div>
        </div>
      )}
    </div>
  )
}

function ManualModelRow({ controller }: Pick<Props, 'controller'>) {
  return (
    <div className={styles.manualRow}>
      <ProductIcon name="plus" aria-hidden="true" />
      <input
        value={controller.manualModel}
        onChange={(event) => controller.setManualModel(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter') {
            event.preventDefault()
            controller.addManualModel()
          }
        }}
        placeholder="Or enter a model ID"
      />
      <button
        type="button"
        onClick={controller.addManualModel}
        disabled={!controller.manualModel.trim()}
      >
        Add
      </button>
    </div>
  )
}

function ModelPicker({ controller }: Pick<Props, 'controller'>) {
  const selectVisible = () => {
    const allSelected = controller.visibleModels.every((model) => controller.selected.has(model))
    controller.setSelected((current) => {
      const next = new Set(current)
      controller.visibleModels.forEach((model) =>
        allSelected ? next.delete(model) : next.add(model),
      )
      return next
    })
  }
  return (
    <section className={styles.modelSection}>
      <div className={styles.modelSectionHeader}>
        <div>
          <h3>Models</h3>
          <span>{controller.selected.size} selected</span>
        </div>
        <div className={styles.modelActions}>
          <div className={styles.modelSearch}>
            <ProductIcon name="search" aria-hidden="true" />
            <input
              type="search"
              value={controller.modelSearch}
              onChange={(event) => controller.setModelSearch(event.target.value)}
              placeholder="Filter models"
            />
          </div>
          <button type="button" onClick={selectVisible}>
            <ProductIcon name="check" aria-hidden="true" />
            Select all
          </button>
        </div>
      </div>
      <div className={styles.modelList}>
        {controller.visibleModels.map((model) => (
          <ModelOption key={model} controller={controller} model={model} />
        ))}
      </div>
    </section>
  )
}

function ModelOption({ controller, model }: Pick<Props, 'controller'> & { model: string }) {
  const logicalName = controller.resolvedModelNames.get(model) ?? model
  const renamed = logicalName !== requestedConnectedModelName(controller.advanced.namePrefix, model)
  const catalogID = controller.catalogModels.get(model)
  const toggle = () =>
    controller.setSelected((current) => {
      const next = new Set(current)
      if (next.has(model)) next.delete(model)
      else next.add(model)
      return next
    })
  return (
    <label className={styles.modelOption}>
      <input type="checkbox" checked={controller.selected.has(model)} onChange={toggle} />
      <span>
        <strong>{logicalName}</strong>
        {catalogID ? (
          <small>Built-in · {controller.modelDisplayNames.get(catalogID)}</small>
        ) : renamed ? (
          <small>Named to avoid a public model conflict</small>
        ) : null}
      </span>
      <ProductIcon className={styles.modelCheck} name="check" aria-hidden="true" />
    </label>
  )
}

function DialogFooter({
  controller,
  onClose,
  onManualSetup,
}: Pick<Props, 'controller' | 'onClose' | 'onManualSetup'>) {
  return (
    <footer className={styles.footer}>
      {controller.stage === 'provider' ? (
        <button
          type="button"
          className={styles.secondaryButton}
          onClick={() => {
            onClose()
            onManualSetup()
          }}
          disabled={controller.busy}
        >
          <ProductIcon name="settings" aria-hidden="true" />
          Manual setup
        </button>
      ) : (
        <span />
      )}
      <div>
        <button
          type="button"
          className={styles.secondaryButton}
          onClick={onClose}
          disabled={controller.busy}
        >
          <ProductIcon name="close" aria-hidden="true" />
          Cancel
        </button>
        {controller.stage === 'models' ? (
          <button
            type="button"
            className={styles.primaryButton}
            onClick={() => void controller.submit()}
            disabled={controller.busy || controller.selected.size === 0}
          >
            <ProductIcon name="plus" aria-hidden="true" />
            {controller.saving
              ? 'Adding…'
              : `Add ${controller.selected.size || ''} model${controller.selected.size === 1 ? '' : 's'}`}
          </button>
        ) : null}
      </div>
    </footer>
  )
}
