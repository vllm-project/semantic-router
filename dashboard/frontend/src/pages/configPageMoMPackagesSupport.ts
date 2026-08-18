import type {
  RecipeActivationState,
  RecipeDescriptor,
  RecipePackageErrorCode,
  RecipePackageWarning,
} from '../types/recipe'
import { RecipeApiError } from '../utils/recipeApi'

export const RECIPE_PACKAGE_PAGE_SIZE = 25

export type ManagedRecipeProtectionState = Extract<
  RecipeActivationState,
  'active' | 'recovering' | 'inconsistent'
>

export function resolveManagedRecipeProtection(
  descriptor: RecipeDescriptor | null,
): ManagedRecipeProtectionState | null {
  if (!descriptor?.managed) return null
  const state = descriptor.activation_state
  if (state === 'active') return descriptor.active_recipe_digest ? state : null
  return state === 'recovering' || state === 'inconsistent' ? state : null
}

export type RecipePackageCapabilityBlocker =
  | 'server_readonly'
  | 'runtime_config_readonly'
  | 'permission_required'
  | null

export interface RecipePackageCapabilities {
  canActivate: boolean
  activationBlocker: RecipePackageCapabilityBlocker
}

export function resolveRecipePackageCapabilities(
  serverReadonly: boolean,
  runtimeConfigWritable: boolean,
  hasDeployPermission: boolean,
): RecipePackageCapabilities {
  const activationBlocker: RecipePackageCapabilityBlocker = serverReadonly
    ? 'server_readonly'
    : !runtimeConfigWritable
      ? 'runtime_config_readonly'
      : !hasDeployPermission
        ? 'permission_required'
        : null
  return {
    canActivate: activationBlocker === null,
    activationBlocker,
  }
}

export type RecipePackageOperation =
  | 'load'
  | 'activate'
  | 'deactivate'
  | 'activate-preview'
  | 'deactivate-preview'

export interface RecipePackageErrorPresentation {
  title: string
  message: string
  code?: RecipePackageErrorCode
}

export interface RecipePackageWarningPresentation {
  code: string
  path: string
  message: string
}

export function presentRecipePackageWarning(
  warning: RecipePackageWarning,
): RecipePackageWarningPresentation {
  if (warning.code === 'environment_binding_required') {
    const environmentName = warning.message.match(
      /environment variable\s+([A-Za-z_][A-Za-z0-9_]*)/i,
    )?.[1]
    return {
      code: warning.code,
      path: warning.path,
      message: environmentName
        ? `Requires environment variable ${environmentName} to be authorized before activation.`
        : 'Requires an authorized environment variable before activation.',
    }
  }
  return {
    code: warning.code,
    path: warning.path,
    message: 'Review this package warning before activation.',
  }
}

const RUNTIME_CONFIG_MUTATION_ERRORS: Record<string, string> = {
  managed_recipe_active:
    'This runtime configuration is controlled by a managed Recipe. Use package activation, repair, or Restore source instead.',
  deploy_in_progress: 'Another runtime configuration operation is in progress. Try again shortly.',
  config_coordination_failed:
    'Runtime configuration coordination is unavailable. Refresh managed Recipe state before retrying.',
  runtime_config_readonly:
    'The runtime configuration mount is read-only. Runtime changes are disabled.',
  readonly_mode: 'This Dashboard is read-only. Configuration editing is disabled.',
}

export function presentRuntimeConfigMutationError(code: string | undefined): string | undefined {
  return code ? RUNTIME_CONFIG_MUTATION_ERRORS[code] : undefined
}

export function formatPackageInstalledAt(value: string): string {
  const timestamp = Date.parse(value)
  if (!Number.isFinite(timestamp)) return 'Install time unavailable'
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(timestamp)
}

const PACKAGE_ERRORS: Partial<Record<RecipePackageErrorCode, RecipePackageErrorPresentation>> = {
  invalid_request: {
    title: 'Package request is invalid',
    message: 'Refresh installed packages and retry the lifecycle action.',
  },
  invalid_recipe: {
    title: 'Recipe schema validation failed',
    message: 'The installed package does not contain a conformant managed Recipe.',
  },
  package_not_found: {
    title: 'Installed package was not found',
    message: 'Refresh installed packages and choose another Recipe to activate.',
  },
  deploy_in_progress: {
    title: 'Another configuration operation is in progress',
    message: 'Wait for the current Router configuration operation to finish, then retry.',
  },
  activation_failed: {
    title: 'Recipe activation failed',
    message: 'The new Router configuration was rejected and the previous Recipe was restored.',
  },
  activation_rollback_failed: {
    title: 'Activation and rollback failed',
    message:
      'The Router could not activate this Recipe or restore the previous configuration. Check Router health before retrying.',
  },
  activation_inconsistent: {
    title: 'Router repair could not start',
    message: 'Refresh activation state, then retry repair with an installed Recipe package.',
  },
  activation_confirmation_required: {
    title: 'Activation plan changed',
    message:
      'Review the current listener, storage, and authentication restart plan before confirming again.',
  },
  environment_binding_required: {
    title: 'Recipe environment binding unavailable',
    message:
      'Authorize and provide the required environment variables when starting the managed stack, then retry activation. The runtime configuration was not changed.',
  },
  activation_incompatible: {
    title: 'Recipe is incompatible with this running stack',
    message:
      'This package requires a different listener or managed-service topology. Start a compatible stack before activating it. The runtime configuration was not changed.',
  },
  managed_storage_isolation_required: {
    title: 'Managed storage needs loopback isolation',
    message:
      'Preserve the existing storage data, recreate the affected managed storage container with loopback-only host port bindings, and retry. No Recipe activation or source restore was started.',
  },
  managed_recipe_active: {
    title: 'Managed Recipe controls this configuration',
    message:
      'Use Recipe package activation, repair, or Restore source instead of an ordinary runtime configuration update.',
  },
  config_coordination_failed: {
    title: 'Runtime configuration coordination failed',
    message: 'Refresh the managed Recipe state before retrying this operation.',
  },
  warnings_not_acknowledged: {
    title: 'Package warnings need acknowledgment',
    message: 'Review the package warnings and confirm activation again.',
  },
  runtime_config_readonly: {
    title: 'Runtime configuration is read-only',
    message:
      'Activation, repair, and source restore require a writable runtime configuration mount.',
  },
  readonly_mode: {
    title: 'Server is read-only',
    message: 'The explicit server-wide read-only policy disables every Recipe package change.',
  },
}

const DEACTIVATION_ERRORS: Partial<Record<RecipePackageErrorCode, RecipePackageErrorPresentation>> =
  {
    activation_incompatible: {
      title: 'Source runtime is incompatible with this running stack',
      message:
        'The preserved source configuration requires a different stack topology. The managed Recipe remains active.',
    },
    activation_failed: {
      title: 'Source runtime restore failed',
      message: 'The source configuration could not be restored. The managed Recipe remains active.',
    },
    activation_rollback_failed: {
      title: 'Source restore and rollback failed',
      message:
        'The Router could not restore the source configuration or return to a consistent managed state. Check Router health before retrying.',
    },
  }

const ACTIVATION_PREVIEW_ERRORS: Partial<
  Record<RecipePackageErrorCode, RecipePackageErrorPresentation>
> = {
  activation_failed: {
    title: 'Activation plan could not be prepared',
    message:
      'The Dashboard could not inspect or prepare the activation plan. Recipe activation was not started.',
  },
}

const DEACTIVATION_PREVIEW_ERRORS: Partial<
  Record<RecipePackageErrorCode, RecipePackageErrorPresentation>
> = {
  activation_failed: {
    title: 'Source restore plan could not be prepared',
    message:
      'The Dashboard could not inspect or prepare the source restore plan. Source restore was not started.',
  },
}

export function presentRecipePackageError(
  reason: unknown,
  operation: RecipePackageOperation,
): RecipePackageErrorPresentation {
  const code = reason instanceof RecipeApiError ? reason.code : undefined
  const known = code
    ? operation === 'deactivate'
      ? DEACTIVATION_ERRORS[code] || PACKAGE_ERRORS[code]
      : operation === 'activate-preview'
        ? ACTIVATION_PREVIEW_ERRORS[code] || PACKAGE_ERRORS[code]
        : operation === 'deactivate-preview'
          ? DEACTIVATION_PREVIEW_ERRORS[code] || DEACTIVATION_ERRORS[code] || PACKAGE_ERRORS[code]
          : PACKAGE_ERRORS[code]
    : undefined
  if (known) return { ...known, code }

  if (operation === 'load') {
    return {
      title: 'Installed packages are unavailable',
      message: 'The Dashboard could not load the installed Recipe package inventory.',
    }
  }
  if (operation === 'activate') {
    return {
      title: 'Recipe activation failed',
      message: 'The Dashboard could not complete the activation request. Refresh before retrying.',
    }
  }
  if (operation === 'activate-preview') {
    return {
      title: 'Activation plan could not be prepared',
      message:
        'The Dashboard could not inspect or prepare the activation plan. Recipe activation was not started.',
    }
  }
  if (operation === 'deactivate-preview') {
    return {
      title: 'Source restore plan could not be prepared',
      message:
        'The Dashboard could not inspect or prepare the source restore plan. Source restore was not started.',
    }
  }
  return {
    title: 'Source runtime restore failed',
    message:
      'The Dashboard could not complete the source restore request. Refresh managed Recipe state before retrying.',
  }
}
