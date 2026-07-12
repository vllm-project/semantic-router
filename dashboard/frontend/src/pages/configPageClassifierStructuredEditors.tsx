import { KeyValueEditor } from '../components/KeyValueEditor'
import { StringListEditor } from '../components/StringListEditor'
import {
  normalizeMcpArguments,
  normalizeMcpEnvironment,
} from './configPageClassifierStructuredSupport'

interface UnknownEditorProps {
  value: unknown
  onChange: (value: unknown) => void
}

export function McpArgumentsEditor({ value, onChange }: UnknownEditorProps) {
  let argumentsList: string[] = []
  try {
    argumentsList = normalizeMcpArguments(value)
  } catch {
    // Keep malformed legacy values editable instead of failing the whole configuration surface.
  }

  return (
    <StringListEditor
      value={argumentsList}
      onChange={onChange}
      addLabel="Add argument"
      emptyLabel="No command-line arguments configured."
      itemLabel="Argument"
      placeholder="--port"
    />
  )
}

export function McpEnvironmentEditor({ value, onChange }: UnknownEditorProps) {
  let environment: Record<string, string> = {}
  try {
    environment = normalizeMcpEnvironment(value)
  } catch {
    // Keep malformed legacy values editable instead of failing the whole configuration surface.
  }

  return (
    <KeyValueEditor
      value={environment}
      onChange={onChange}
      addLabel="Add environment variable"
      emptyLabel="No environment variables configured."
      keyLabel="Variable"
      keyPlaceholder="API_KEY"
      valueLabel="Value"
      valuePlaceholder="Environment value"
    />
  )
}
