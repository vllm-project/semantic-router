import type {
  ConfigVersion,
  DeployResult,
  DeployStep,
  Diagnostic,
} from '@/types/dsl'
import {
  activateConfigRevision,
  createAndActivateConfigRevision,
  formatConfigRevisionLabel,
  listConfigVersionHistory,
} from '@/utils/configRevisionApi'

interface DeployStatusService {
  name?: string
  healthy?: boolean
}

interface DeployStatusResponse {
  overall?: string
  services?: DeployStatusService[]
}

export interface DSLDeployState {
  deploying: boolean
  deployStep: DeployStep | null
  deployResult: DeployResult | null
  showDeployConfirm: boolean
  configVersions: ConfigVersion[]
  deployPreviewCurrent: string
  deployPreviewMerged: string
  deployPreviewLoading: boolean
  deployPreviewError: string | null
}

interface DSLDeployContext extends DSLDeployState {
  dslSource: string
  yamlOutput: string
  wasmReady: boolean
  dirty: boolean
  diagnostics: Diagnostic[]
  compile(): void
}

interface DSLDeployMutations {
  dirty?: boolean
}

type DSLDeploySet = (partial: Partial<DSLDeployState> & DSLDeployMutations) => void
type DSLDeployGet = () => DSLDeployContext

export interface DSLDeployActions {
  requestDeploy(): void
  executeDeploy(): Promise<void>
  dismissDeploy(): void
  rollback(version: string): Promise<void>
  fetchVersions(): Promise<void>
}

export const initialDSLDeployState: DSLDeployState = {
  deploying: false,
  deployStep: null,
  deployResult: null,
  showDeployConfirm: false,
  configVersions: [],
  deployPreviewCurrent: '',
  deployPreviewMerged: '',
  deployPreviewLoading: false,
  deployPreviewError: null,
}

async function waitForRuntimeHealth(): Promise<boolean> {
  for (let i = 0; i < 10; i++) {
    await new Promise(r => setTimeout(r, 500))
    try {
      const statusResp = await fetch('/api/status')
      if (!statusResp.ok) {
        continue
      }

      const statusData = await statusResp.json() as DeployStatusResponse
      const routerHealthy = statusData.services?.find(service => service.name === 'Router')?.healthy === true
      const envoyService = statusData.services?.find(service => service.name === 'Envoy')
      const envoyHealthy = envoyService ? envoyService.healthy === true : true

      if (statusData.overall === 'healthy' && routerHealthy && envoyHealthy) {
        return true
      }
    } catch {
      // continue polling
    }
  }

  return false
}

async function fetchDeployPreview(yaml: string): Promise<{ current: string; preview: string }> {
  const response = await fetch('/api/router/config/deploy/preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ yaml }),
  })

  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    throw new Error(data.message || data.error || 'Failed to fetch preview')
  }

  return response.json()
}

export function createDSLDeployActions(
  set: DSLDeploySet,
  get: DSLDeployGet,
): DSLDeployActions {
  const fetchVersions = async () => {
    try {
      const versions = await listConfigVersionHistory()
      set({ configVersions: versions })
    } catch {
      // silently fail
    }
  }

  const dismissDeploy = () => {
    set({
      showDeployConfirm: false,
      deployResult: null,
      deployStep: null,
      deployPreviewCurrent: '',
      deployPreviewMerged: '',
      deployPreviewLoading: false,
      deployPreviewError: null,
    })
  }

  return {
    requestDeploy() {
      const { yamlOutput, dslSource, wasmReady, dirty } = get()
      if (!wasmReady || !dslSource.trim()) return

      if (!yamlOutput || dirty) {
        get().compile()
      }

      const { diagnostics, yamlOutput: compiledYAML } = get()
      const hasErrors = diagnostics.some(diagnostic => diagnostic.level === 'error')
      if (hasErrors || !compiledYAML) {
        set({
          deployResult: {
            status: 'error',
            message: 'Cannot deploy: DSL has compilation errors. Fix errors and compile first.',
          },
          showDeployConfirm: false,
        })
        return
      }

      set({
        showDeployConfirm: true,
        deployResult: null,
        deployPreviewCurrent: '',
        deployPreviewMerged: '',
        deployPreviewLoading: true,
        deployPreviewError: null,
      })

      fetchDeployPreview(compiledYAML)
        .then((preview) => {
          set({
            deployPreviewCurrent: preview.current,
            deployPreviewMerged: preview.preview,
            deployPreviewLoading: false,
          })
        })
        .catch((err) => {
          set({
            deployPreviewLoading: false,
            deployPreviewError: err instanceof Error ? err.message : String(err),
          })
        })
    },

    async executeDeploy() {
      const { yamlOutput, deployPreviewMerged, dslSource } = get()
      const runtimeConfigYAML = deployPreviewMerged || yamlOutput
      if (!runtimeConfigYAML) return

      console.log('[dslStore.executeDeploy] Sending deploy: YAML size=%d, DSL size=%d', yamlOutput.length, dslSource.length)

      set({ deploying: true, deployStep: 'validating', showDeployConfirm: false, deployResult: null })

      try {
        set({ deployStep: 'backing_up' })
        await new Promise(r => setTimeout(r, 200))

        set({ deployStep: 'writing' })
        const activatedRevision = await createAndActivateConfigRevision({
          runtimeConfigYAML,
          source: 'builder_revision_deploy',
          summary: 'Applied config deploy from Builder',
          dslSource,
          metadata: {
            ui_surface: 'builder',
            dsl_present: dslSource.trim() !== '',
          },
        })

        set({ deployStep: 'reloading' })
        const healthy = await waitForRuntimeHealth()
        const revisionLabel = formatConfigRevisionLabel(activatedRevision.id)

        set({
          deploying: false,
          deployStep: 'done',
          deployResult: {
            status: 'success',
            version: activatedRevision.id,
            message: healthy
              ? `Activated revision ${revisionLabel} — Router and Envoy reloaded successfully.`
              : `Activated revision ${revisionLabel} — Runtime reload status unknown (check logs).`,
          },
          dirty: false,
        })

        void fetchVersions()
        window.dispatchEvent(new CustomEvent('config-deployed'))
      } catch (err) {
        set({
          deploying: false,
          deployStep: 'error',
          deployResult: {
            status: 'error',
            message: `Deploy failed: ${err instanceof Error ? err.message : String(err)}`,
          },
        })
      }
    },

    dismissDeploy,

    async rollback(version: string) {
      set({ deploying: true, deployStep: 'writing', deployResult: null })

      try {
        const activatedRevision = await activateConfigRevision(version)
        set({ deployStep: 'reloading' })
        const healthy = await waitForRuntimeHealth()
        const revisionLabel = formatConfigRevisionLabel(activatedRevision.id)

        set({
          deploying: false,
          deployStep: 'done',
          deployResult: {
            status: 'success',
            version: activatedRevision.id,
            message: healthy
              ? `Reactivated revision ${revisionLabel} — Router and Envoy reloaded successfully.`
              : `Reactivated revision ${revisionLabel} — Runtime reload status unknown (check logs).`,
          },
        })

        void fetchVersions()
      } catch (err) {
        set({
          deploying: false,
          deployStep: 'error',
          deployResult: {
            status: 'error',
            message: `Rollback failed: ${err instanceof Error ? err.message : String(err)}`,
          },
        })
      }
    },

    fetchVersions,
  }
}
