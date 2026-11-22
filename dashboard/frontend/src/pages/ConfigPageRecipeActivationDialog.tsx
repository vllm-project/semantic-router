import { useEffect, useId, useState } from 'react'
import { Info, Minus, Package, Plus, Power, RefreshCw, TriangleAlert, Zap } from 'lucide-react'

import type {
  RecipeActivationPlan,
  RecipePackageActivationResult,
  RecipePackageDeactivationResult,
  RecipePackageSummary,
} from '../types/recipe'
import {
  Alert,
  Badge,
  Button,
  Checkbox,
  CopyField,
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  IconTile,
  Label,
  SectionLabel,
  Skeleton,
  cn,
} from '../ui'
import { planChanges, restartRequired, type PlanChange } from './configPageRecipesPageSupport'
import {
  RecipeApiError,
  activateRecipePackage,
  deactivateRecipePackage,
  previewRecipePackageActivation,
  previewRecipePackageDeactivation,
} from '../utils/recipeApi'

export type RecipeToggleMode = 'activate' | 'deactivate'

export interface RecipeToggleTarget {
  mode: RecipeToggleMode
  recipe: RecipePackageSummary
}

export type RecipeToggleResult = RecipePackageActivationResult | RecipePackageDeactivationResult

function ChangeRow({ change }: { change: PlanChange }) {
  const Icon = change.kind === 'add' ? Plus : change.kind === 'remove' ? Minus : Info
  return (
    <li className="flex items-start gap-2.5 py-1.5 text-[13px]">
      <span
        className={cn(
          'mt-px grid size-5 shrink-0 place-items-center rounded-full',
          change.kind === 'add' && 'bg-success-soft text-success-foreground',
          change.kind === 'remove' && 'bg-destructive-soft text-destructive-foreground',
          change.kind === 'info' && 'bg-secondary text-muted-foreground',
        )}
      >
        <Icon className="size-3" strokeWidth={2.5} />
      </span>
      <span className="leading-5">{change.text}</span>
    </li>
  )
}

/**
 * Confirms activating or deactivating a recipe package: it shows the preview
 * plan, says whether the change hot-switches or needs a restart, and binds the
 * confirmation to that exact plan.
 */
export default function ConfigPageRecipeActivationDialog({
  target,
  onOpenChange,
  onDone,
}: {
  target: RecipeToggleTarget | null
  onOpenChange: (open: boolean) => void
  onDone: (target: RecipeToggleTarget, result: RecipeToggleResult) => void
}) {
  const acknowledgeId = useId()
  const [plan, setPlan] = useState<RecipeActivationPlan | null>(null)
  const [planError, setPlanError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [acknowledged, setAcknowledged] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [attempt, setAttempt] = useState(0)
  const warnings = target?.mode === 'activate' ? target.recipe.warnings : []
  const needsAcknowledgement = warnings.length > 0

  useEffect(() => {
    setPlan(null)
    setPlanError(null)
    setAcknowledged(false)
    setError(null)
    setBusy(false)
  }, [target])

  useEffect(() => {
    if (!target || (needsAcknowledgement && !acknowledged)) return
    const controller = new AbortController()
    setLoading(true)
    setPlanError(null)
    const request =
      target.mode === 'activate'
        ? previewRecipePackageActivation(
            target.recipe.recipe_digest,
            needsAcknowledgement,
            controller.signal,
          )
        : previewRecipePackageDeactivation(controller.signal)
    void request
      .then((next) => {
        if (!controller.signal.aborted) setPlan(next)
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return
        setPlan(null)
        setPlanError(
          reason instanceof Error ? reason.message : 'The change could not be previewed.',
        )
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [acknowledged, attempt, needsAcknowledgement, target])

  const confirm = async () => {
    if (!target || !plan) return
    setBusy(true)
    setError(null)
    const confirmation = plan.requires_confirmation
      ? {
          expected_plan_digest: plan.plan_digest,
          confirm_stack_recreation: plan.mode === 'stack_recreation',
        }
      : undefined
    try {
      const result =
        target.mode === 'activate'
          ? await activateRecipePackage(
              target.recipe.recipe_digest,
              needsAcknowledgement,
              undefined,
              confirmation,
            )
          : await deactivateRecipePackage(undefined, confirmation)
      onDone(target, result)
    } catch (reason) {
      if (reason instanceof RecipeApiError && reason.code === 'activation_confirmation_required') {
        setError(
          'The plan changed since it was previewed. Review the new plan, then confirm again.',
        )
        setAttempt((current) => current + 1)
      } else {
        setError(reason instanceof Error ? reason.message : 'The change was not applied.')
      }
    } finally {
      setBusy(false)
    }
  }

  if (!target) return null
  const { mode, recipe } = target
  const changes = plan ? planChanges(plan) : []
  const restart = restartRequired(plan)
  const waitingForAcknowledgement = needsAcknowledgement && !acknowledged

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !busy) onOpenChange(false)
      }}
    >
      <DialogContent size="md" aria-busy={busy || loading} className="flex flex-col">
        <DialogHeader>
          <div className="mb-2 flex items-center gap-2.5">
            <IconTile icon={Package} tone="blue" size="sm" />
            <span className="font-mono text-xs text-muted-foreground">
              {recipe.id} · {recipe.version}
            </span>
          </div>
          <DialogTitle>
            {mode === 'activate' ? 'Activate' : 'Deactivate'}{' '}
            <span className="font-mono">{recipe.name || recipe.id}</span>?
          </DialogTitle>
          <DialogDescription>
            {mode === 'activate'
              ? 'The Router loads the package’s configuration and publishes its public models.'
              : 'The Router returns to the configuration it ran before the package; the package’s public models stop resolving.'}
          </DialogDescription>
        </DialogHeader>
        <DialogBody className="flex min-h-0 flex-1 flex-col gap-4 pb-5">
          {needsAcknowledgement ? (
            <section className="flex flex-col gap-2">
              <SectionLabel>Package warnings · {warnings.length}</SectionLabel>
              <ul className="flex flex-col gap-1.5">
                {warnings.map((warning) => (
                  <li
                    key={`${warning.code}:${warning.path}`}
                    className="flex items-start gap-2 rounded-md bg-warning-soft px-2.5 py-2 text-xs"
                  >
                    <TriangleAlert className="mt-px size-3.5 shrink-0 text-warning-foreground" />
                    <span className="flex min-w-0 flex-col">
                      <span className="text-foreground/90">{warning.message}</span>
                      <span className="font-mono text-[11px] text-muted-foreground">
                        {warning.code} · {warning.path}
                      </span>
                    </span>
                  </li>
                ))}
              </ul>
              <div className="flex items-center gap-2">
                <Checkbox
                  id={acknowledgeId}
                  checked={acknowledged}
                  disabled={busy}
                  onCheckedChange={(checked) => setAcknowledged(checked === true)}
                />
                <Label htmlFor={acknowledgeId} className="font-normal">
                  I reviewed these warnings
                </Label>
              </div>
            </section>
          ) : null}

          {waitingForAcknowledgement ? null : loading ? (
            <div className="flex flex-col gap-2" aria-label="Loading the activation preview">
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-4/5" />
              <Skeleton className="h-16 w-full" />
            </div>
          ) : planError ? (
            <div role="alert">
              <Alert
                tone="danger"
                title="The change could not be previewed"
                actions={
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setAttempt((value) => value + 1)}
                  >
                    <RefreshCw />
                    Retry
                  </Button>
                }
              >
                {planError}
              </Alert>
            </div>
          ) : plan ? (
            <>
              <section>
                <div className="mb-1 flex items-center justify-between gap-2">
                  <SectionLabel>What changes</SectionLabel>
                  <Badge size="sm" tone={restart ? 'warning' : 'info'}>
                    {restart ? 'Stack recreation' : 'Hot switch'}
                  </Badge>
                </div>
                {changes.length ? (
                  <ul className="flex flex-col">
                    {changes.map((change) => (
                      <ChangeRow key={`${change.kind}:${change.text}`} change={change} />
                    ))}
                  </ul>
                ) : (
                  <p className="text-[13px] text-muted-foreground">
                    No listener, storage or management changes; only the routing configuration
                    changes.
                  </p>
                )}
              </section>
              {restart ? (
                <Alert tone="warning" title="Restart required: run vllm-sr serve to apply">
                  <span className="flex flex-col gap-2.5">
                    <span>
                      This change recreates the managed stack. It is saved now and applies when
                      vllm-sr serve creates the containers anew; vllm-sr status reports it until
                      then.
                    </span>
                    <CopyField value="vllm-sr serve" className="bg-surface" />
                  </span>
                </Alert>
              ) : (
                <Alert tone="info" icon={<Zap />} title="Hot switch, no restart">
                  The Router reloads the configuration in place. Requests in flight finish on the
                  current configuration.
                </Alert>
              )}
            </>
          ) : null}
          {error ? (
            <div role="alert">
              <Alert tone="danger" title="The change was not applied">
                {error}
              </Alert>
            </div>
          ) : null}
        </DialogBody>
        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={busy}>
            Cancel
          </Button>
          <Button
            variant={mode === 'deactivate' ? 'destructive' : 'default'}
            loading={busy}
            disabled={!plan || loading || waitingForAcknowledgement}
            onClick={() => void confirm()}
          >
            {busy ? null : <Power />}
            {mode === 'activate'
              ? restart
                ? 'Save and activate on restart'
                : 'Activate'
              : restart
                ? 'Save and deactivate on restart'
                : 'Deactivate'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
