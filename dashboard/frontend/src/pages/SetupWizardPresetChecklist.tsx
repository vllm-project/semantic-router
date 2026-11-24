import { Check, Circle, CircleCheck } from 'lucide-react'
import { Alert, Badge, Button, Card, Progress } from '../ui'
import type {
  ImportedSetupConfig,
  PresetDelta,
  PresetRequestState,
  SetupConfigCounts,
} from './setupWizardSupport'

interface SetupWizardPresetChecklistProps {
  presetDelta: PresetDelta
  presetImportedConfig: ImportedSetupConfig | null
  counts: SetupConfigCounts
  presetError: string | null
  presetRequestState: PresetRequestState
  onImportPresetConfig: () => void
}

export default function SetupWizardPresetChecklist({
  presetDelta,
  presetImportedConfig,
  counts,
  presetError,
  presetRequestState,
  onImportPresetConfig,
}: SetupWizardPresetChecklistProps) {
  const total = presetDelta.configured_models.length + presetDelta.missing_models.length
  const readyCount = presetDelta.configured_models.length
  const isImporting = presetRequestState === 'importing'

  return (
    <Card className="flex flex-col gap-3 p-4" aria-busy={isImporting}>
      <div className="flex items-center justify-between gap-3">
        <span className="text-[13px] font-semibold">
          {presetDelta.ready ? 'Ready to activate' : 'Model requirements'}
        </span>
        <span className="text-xs tabular text-muted-foreground">
          {readyCount}/{total} models ready
        </span>
      </div>
      <Progress
        value={(readyCount / Math.max(1, total)) * 100}
        tone={presetDelta.ready ? 'success' : 'brand'}
        aria-label={`${readyCount} of ${total} required models configured`}
      />
      <ul className="flex flex-col overflow-hidden rounded-lg border border-border">
        {presetDelta.configured_models.map((name) => (
          <li
            key={name}
            className="flex items-center gap-2.5 border-b border-border px-3 py-2 text-[13px] last:border-0"
          >
            <Check className="size-4 shrink-0 text-success" strokeWidth={2.4} aria-hidden />
            <span className="min-w-0 flex-1 truncate font-mono text-[12.5px]">{name}</span>
            <span className="text-xs text-subtle-foreground">configured</span>
          </li>
        ))}
        {presetDelta.missing_models.map((model) => (
          <li
            key={model.name}
            className="flex items-center gap-2.5 border-b border-border px-3 py-2 text-[13px] last:border-0"
          >
            <Circle className="size-4 shrink-0 text-subtle-foreground" aria-hidden />
            <span className="min-w-0 flex-1 truncate font-mono text-[12.5px]">{model.name}</span>
            <Badge size="sm" tone="outline">
              {model.role}
            </Badge>
            <span className="text-xs text-subtle-foreground">add in step 1</span>
          </li>
        ))}
      </ul>
      {!presetDelta.ready ? (
        <p className="text-xs text-muted-foreground">
          Missing models use placeholder endpoints. Update them after activation on the Backend
          models page.
        </p>
      ) : null}
      {presetError ? (
        <p className="text-xs text-destructive-foreground" role="alert">
          {presetError}
        </p>
      ) : null}
      {!presetImportedConfig ? (
        <Button
          variant="outline"
          size="sm"
          className="w-fit"
          onClick={onImportPresetConfig}
          loading={isImporting}
        >
          {isImporting
            ? 'Importing preset…'
            : presetError
              ? 'Retry preset import'
              : presetDelta.ready
                ? 'Import preset config'
                : 'Import with placeholders'}
        </Button>
      ) : (
        <div role="status">
          <Alert tone="success" title="Preset config ready" icon={<CircleCheck />}>
            {counts.models} models · {counts.decisions} decisions · {counts.signals} signals
          </Alert>
        </div>
      )}
    </Card>
  )
}
