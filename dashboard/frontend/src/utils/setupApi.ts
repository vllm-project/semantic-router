import { createAndActivateConfigRevision } from './configRevisionApi'
import { SetupActivateResponse, SetupState, SetupValidateResponse } from '../types/setup'

async function readErrorMessage(response: Response): Promise<string> {
  const body = await response.text()
  if (!body) {
    return `HTTP ${response.status}: ${response.statusText}`
  }

  try {
    const parsed = JSON.parse(body) as { error?: string; message?: string }
    return parsed.message || parsed.error || body
  } catch {
    return body
  }
}

async function postSetupConfig<T>(path: string, config: Record<string, unknown>): Promise<T> {
  const response = await fetch(path, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ config }),
  })

  if (!response.ok) {
    throw new Error(await readErrorMessage(response))
  }

  return response.json()
}

export async function fetchSetupState(): Promise<SetupState> {
  const response = await fetch('/api/setup/state')
  if (!response.ok) {
    throw new Error(await readErrorMessage(response))
  }
  return response.json()
}

export async function validateSetupConfig(
  config: Record<string, unknown>,
): Promise<SetupValidateResponse> {
  return postSetupConfig<SetupValidateResponse>('/api/setup/validate', config)
}

export async function activateSetupConfig(
  config: Record<string, unknown>,
): Promise<SetupActivateResponse> {
  const result = await createAndActivateConfigRevision({
    document: config,
    source: 'setup_wizard_activate',
    summary: 'Activated setup config from Setup Wizard',
    metadata: {
      ui_surface: 'setup_wizard',
      setup_flow: true,
    },
  })

  return {
    status: 'success',
    setupMode: true,
    message: result.message,
  }
}
