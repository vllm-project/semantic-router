import type { ConfigVersion } from '@/types/dsl'
import type { DSLActions } from './dslStoreTypes'
import type {
  DeployStatusResponse,
  DSLStoreGetter,
  DSLStoreSetter,
} from './dslStoreSupport'

interface DeployPreviewResponse {
  current: string
  preview: string
}

interface DeployMutationResponse {
  version?: string
  message?: string
  error?: string
}

function wait(ms: number) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms)
  })
}

async function parseDeployMutationResponse(
  response: Response,
): Promise<DeployMutationResponse> {
  const responseText = await response.text()
  if (!responseText) {
    return {}
  }

  try {
    return JSON.parse(responseText) as DeployMutationResponse
  } catch {
    return { message: responseText }
  }
}

async function fetchDeployPreview(
  set: DSLStoreSetter,
  yaml: string,
  baseYaml: string,
) {
  try {
    const response = await fetch('/api/router/config/deploy/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ yaml, baseYaml }),
    })

    if (!response.ok) {
      const errorPayload = await response.json().catch(
        () => ({}) as Record<string, string>,
      )
      throw new Error(
        errorPayload.message || errorPayload.error || 'Failed to fetch preview',
      )
    }

    const preview = (await response.json()) as DeployPreviewResponse
    set({
      deployPreviewCurrent: preview.current,
      deployPreviewMerged: preview.preview,
      deployPreviewLoading: false,
    })
  } catch (err) {
    set({
      deployPreviewLoading: false,
      deployPreviewError: err instanceof Error ? err.message : String(err),
    })
  }
}

async function pollRuntimeHealth() {
  for (let attempt = 0; attempt < 10; attempt += 1) {
    await wait(500)

    try {
      const statusResponse = await fetch('/api/status')
      if (!statusResponse.ok) {
        continue
      }

      const statusData = (await statusResponse.json()) as DeployStatusResponse
      const routerHealthy =
        statusData.services?.find((service) => service.name === 'Router')
          ?.healthy === true
      const envoyService = statusData.services?.find(
        (service) => service.name === 'Envoy',
      )
      const envoyHealthy = envoyService ? envoyService.healthy === true : true

      if (statusData.overall === 'healthy' && routerHealthy && envoyHealthy) {
        return true
      }
    } catch {
      // Keep polling until the retry budget is exhausted.
    }
  }

  return false
}

export function createDeployActions(
  set: DSLStoreSetter,
  get: DSLStoreGetter,
): Pick<
  DSLActions,
  'requestDeploy' | 'executeDeploy' | 'dismissDeploy' | 'rollback' | 'fetchVersions'
> {
  return {
    requestDeploy() {
      const { yamlOutput, dslSource, wasmReady, dirty, baseConfigYaml } = get()
      if (!wasmReady || !dslSource.trim()) {
        return
      }

      if (!yamlOutput || dirty) {
        get().compile()
      }

      const { diagnostics, yamlOutput: compiledYaml } = get()
      const hasErrors = diagnostics.some((diagnostic) => diagnostic.level === 'error')
      if (hasErrors || !compiledYaml) {
        set({
          deployResult: {
            status: 'error',
            message:
              'Cannot deploy: DSL has compilation errors. Fix errors and compile first.',
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

      void fetchDeployPreview(set, compiledYaml, baseConfigYaml)
    },

    async executeDeploy() {
      const { yamlOutput, dslSource, baseConfigYaml } = get()
      if (!yamlOutput) {
        return
      }

      console.log(
        '[dslStore.executeDeploy] Sending deploy: YAML size=%d, DSL size=%d',
        yamlOutput.length,
        dslSource.length,
      )

      set({
        deploying: true,
        deployStep: 'validating',
        showDeployConfirm: false,
        deployResult: null,
      })

      try {
        set({ deployStep: 'backing_up' })
        await wait(200)

        set({ deployStep: 'writing' })
        const response = await fetch('/api/router/config/deploy', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            yaml: yamlOutput,
            dsl: dslSource,
            baseYaml: baseConfigYaml,
          }),
        })

        const data = await parseDeployMutationResponse(response)
        if (!response.ok) {
          set({
            deploying: false,
            deployStep: 'error',
            deployResult: {
              status: 'error',
              message: data.message || data.error || 'Deploy failed',
            },
          })
          return
        }

        set({ deployStep: 'reloading' })
        const healthy = await pollRuntimeHealth()

        set({
          deploying: false,
          deployStep: 'done',
          deployResult: {
            status: 'success',
            version: data.version,
            message: healthy
              ? `Deployed v${data.version} — Router and Envoy reloaded successfully.`
              : `Deployed v${data.version} — Runtime reload status unknown (check logs).`,
          },
          dirty: false,
        })

        void get().fetchVersions()
        window.dispatchEvent(new CustomEvent('config-deployed'))
      } catch (err) {
        set({
          deploying: false,
          deployStep: 'error',
          deployResult: {
            status: 'error',
            message: `Deploy failed: ${
              err instanceof Error ? err.message : String(err)
            }`,
          },
        })
      }
    },

    dismissDeploy() {
      set({
        showDeployConfirm: false,
        deployResult: null,
        deployStep: null,
        deployPreviewCurrent: '',
        deployPreviewMerged: '',
        deployPreviewLoading: false,
        deployPreviewError: null,
      })
    },

    async rollback(version: string) {
      set({ deploying: true, deployStep: 'writing', deployResult: null })

      try {
        const response = await fetch('/api/router/config/rollback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ version }),
        })

        const data = (await response.json()) as DeployMutationResponse
        if (!response.ok) {
          set({
            deploying: false,
            deployStep: 'error',
            deployResult: {
              status: 'error',
              message: data.message || 'Rollback failed',
            },
          })
          return
        }

        set({ deployStep: 'reloading' })
        await wait(2000)

        set({
          deploying: false,
          deployStep: 'done',
          deployResult: {
            status: 'success',
            version: data.version,
            message: `Rolled back to v${data.version}. Router will reload automatically.`,
          },
        })

        void get().fetchVersions()
      } catch (err) {
        set({
          deploying: false,
          deployStep: 'error',
          deployResult: {
            status: 'error',
            message: `Rollback failed: ${
              err instanceof Error ? err.message : String(err)
            }`,
          },
        })
      }
    },

    async fetchVersions() {
      try {
        const response = await fetch('/api/router/config/versions')
        if (!response.ok) {
          return
        }

        const versions = (await response.json()) as ConfigVersion[]
        set({ configVersions: versions || [] })
      } catch {
        // Ignore background refresh failures.
      }
    },
  }
}
