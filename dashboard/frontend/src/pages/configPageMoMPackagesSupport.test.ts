import { describe, expect, it } from 'vitest'

import { RecipeApiError } from '../utils/recipeApi'
import type { RecipeDescriptor } from '../types/recipe'
import {
  presentRecipePackageError,
  presentRecipePackageWarning,
  presentRuntimeConfigMutationError,
  resolveManagedRecipeProtection,
  resolveRecipePackageCapabilities,
} from './configPageMoMPackagesSupport'

describe('Recipe package management support', () => {
  it.each(['active', 'recovering', 'inconsistent'] as const)(
    'protects ordinary config editors while managed activation state is %s',
    (activationState) => {
      expect(
        resolveManagedRecipeProtection({
          managed: true,
          activation_state: activationState,
          ...(activationState === 'active' ? { active_recipe_digest: 'sha256:active' } : {}),
        } as RecipeDescriptor),
      ).toBe(activationState)
    },
  )

  it('does not protect ordinary editors for unmanaged or inactive descriptors', () => {
    expect(
      resolveManagedRecipeProtection({
        managed: false,
        activation_state: 'active',
      } as RecipeDescriptor),
    ).toBeNull()
    expect(
      resolveManagedRecipeProtection({
        managed: true,
        activation_state: 'none',
      } as RecipeDescriptor),
    ).toBeNull()
    expect(
      resolveManagedRecipeProtection({
        managed: true,
        activation_state: 'active',
      } as RecipeDescriptor),
    ).toBeNull()
  })

  it.each([
    [
      false,
      true,
      true,
      {
        canActivate: true,
        activationBlocker: null,
      },
    ],
    [
      false,
      true,
      false,
      {
        canActivate: false,
        activationBlocker: 'permission_required',
      },
    ],
    [
      true,
      true,
      true,
      {
        canActivate: false,
        activationBlocker: 'server_readonly',
      },
    ],
    [
      false,
      false,
      true,
      {
        canActivate: false,
        activationBlocker: 'runtime_config_readonly',
      },
    ],
  ] as const)(
    'resolves lifecycle capability (serverReadonly=%s, runtimeWritable=%s, deploy=%s)',
    (serverReadonly, runtimeWritable, canDeploy, expected) => {
      expect(resolveRecipePackageCapabilities(serverReadonly, runtimeWritable, canDeploy)).toEqual(
        expected,
      )
    },
  )

  it.each([
    ['invalid_recipe', 'Recipe schema validation failed'],
    ['activation_rollback_failed', 'Activation and rollback failed'],
    ['environment_binding_required', 'Recipe environment binding unavailable'],
    ['activation_incompatible', 'Recipe is incompatible with this running stack'],
    ['managed_storage_isolation_required', 'Managed storage needs loopback isolation'],
    ['managed_recipe_active', 'Managed Recipe controls this configuration'],
    ['runtime_config_readonly', 'Runtime configuration is read-only'],
    ['readonly_mode', 'Server is read-only'],
  ] as const)('distinguishes the %s failure category', (code, title) => {
    const error = new RecipeApiError('private backend detail', 422, code)

    const presentation = presentRecipePackageError(error, 'activate')
    expect(presentation.title).toBe(title)
    expect(`${presentation.title} ${presentation.message}`).not.toContain('private backend detail')
  })

  it('shows only an environment binding name and JSON Pointer, never warning values', () => {
    const presentation = presentRecipePackageWarning({
      code: 'environment_binding_required',
      path: '/providers/models/0/api_key',
      message:
        'Recipe requires environment variable ROUTER_API_KEY to be explicitly authorized before activation; value=super-secret.',
    })

    expect(presentation).toMatchObject({
      code: 'environment_binding_required',
      path: '/providers/models/0/api_key',
    })
    expect(presentation.message).toContain('ROUTER_API_KEY')
    expect(JSON.stringify(presentation)).not.toContain('super-secret')
  })

  it('never echoes arbitrary advisory warning messages', () => {
    const presentation = presentRecipePackageWarning({
      code: 'future_advisory',
      path: '/global/example',
      message: 'value=should-never-render',
    })

    expect(presentation.message).toBe('Review this package warning before activation.')
    expect(JSON.stringify(presentation)).not.toContain('should-never-render')
  })

  it.each([
    ['activation_failed', 'Source runtime restore failed'],
    ['activation_rollback_failed', 'Source restore and rollback failed'],
    ['activation_incompatible', 'Source runtime is incompatible with this running stack'],
  ] as const)('presents %s in source-restore terms', (code, title) => {
    const presentation = presentRecipePackageError(
      new RecipeApiError('backend detail must not render', 409, code),
      'deactivate',
    )

    expect(presentation.title).toBe(title)
    expect(`${presentation.title} ${presentation.message}`).not.toContain('backend detail')
  })

  it.each([
    [
      'activate-preview',
      'Activation plan could not be prepared',
      'Recipe activation was not started',
    ],
    [
      'deactivate-preview',
      'Source restore plan could not be prepared',
      'Source restore was not started',
    ],
  ] as const)(
    'uses non-mutating language for an unknown %s failure',
    (operation, title, message) => {
      const presentation = presentRecipePackageError(
        new RecipeApiError('backend detail must not render', 500, 'activation_failed'),
        operation,
      )

      expect(presentation).toMatchObject({ title })
      expect(presentation.message).toContain(message)
      expect(`${presentation.title} ${presentation.message}`.toLowerCase()).not.toContain(
        'restored',
      )
      expect(`${presentation.title} ${presentation.message}`.toLowerCase()).not.toContain(
        'rollback',
      )
    },
  )

  it.each(['activate-preview', 'deactivate-preview'] as const)(
    'shows safe loopback remediation for %s without backend details',
    (operation) => {
      const presentation = presentRecipePackageError(
        new RecipeApiError(
          'managed redis storage 6379 is exposed at 0.0.0.0; private-container-marker',
          409,
          'managed_storage_isolation_required',
        ),
        operation,
      )

      expect(presentation).toMatchObject({
        code: 'managed_storage_isolation_required',
        title: 'Managed storage needs loopback isolation',
      })
      expect(presentation.message).toContain('Preserve')
      expect(presentation.message).toContain('loopback-only')
      expect(JSON.stringify(presentation)).not.toMatch(
        /redis|6379|0\.0\.0\.0|private-container-marker/,
      )
      expect(presentation.message.toLowerCase()).not.toContain('restored')
    },
  )

  it('uses safe managed-runtime mutation copy for stable error codes', () => {
    expect(presentRuntimeConfigMutationError('managed_recipe_active')).toContain(
      'controlled by a managed Recipe',
    )
    expect(presentRuntimeConfigMutationError('config_coordination_failed')).toContain(
      'coordination is unavailable',
    )
  })
})
