import React, { useId, useState } from 'react'
import { ChevronDown, Plus } from 'lucide-react'

import { Badge, Button, Input, cn } from '../ui'
import NativeSelect from './NativeSelect'

export interface Endpoint {
  name: string
  endpoint: string
  protocol: 'http' | 'https'
  weight: number
}

interface EndpointsEditorProps {
  endpoints: Endpoint[]
  onChange: (endpoints: Endpoint[]) => void
}

const getProtocol = (protocol: string | undefined): 'http' | 'https' =>
  protocol === 'https' ? 'https' : 'http'

const EndpointsEditor: React.FC<EndpointsEditorProps> = ({ endpoints, onChange }) => {
  const editorId = useId()
  const [editingIndex, setEditingIndex] = useState<number | null>(null)

  const handleAdd = () => {
    const newEndpoint: Endpoint = {
      name: `endpoint-${endpoints.length + 1}`,
      endpoint: 'localhost:8000',
      protocol: 'http',
      weight: 1,
    }
    onChange([...endpoints, newEndpoint])
    setEditingIndex(endpoints.length)
  }

  const handleUpdate = (index: number, field: keyof Endpoint, value: string | number) => {
    const updated = [...endpoints]
    updated[index] = { ...updated[index], [field]: value }
    onChange(updated)
  }

  const handleDelete = (index: number) => {
    onChange(endpoints.filter((_, i) => i !== index))
    if (editingIndex === index) {
      setEditingIndex(null)
    }
  }

  return (
    <div className="flex flex-col gap-2.5">
      {endpoints.length > 0 ? (
        <div className="flex flex-col gap-2">
          {endpoints.map((ep, index) => {
            const protocol = getProtocol(ep.protocol)
            const name = ep.name || `endpoint-${index + 1}`
            const address = ep.endpoint || ''
            const weight = Number.isFinite(ep.weight) ? ep.weight : 1
            const editing = editingIndex === index
            const fieldId = (field: string) => `${editorId}-${index}-${field}`

            return (
              <div
                key={index}
                className="overflow-hidden rounded-lg border border-border bg-surface shadow-xs"
              >
                <div className="flex items-center justify-between gap-3 px-3.5 py-2.5">
                  <div className="flex min-w-0 flex-col gap-1">
                    <span className="truncate font-mono text-[13px] font-medium">{name}</span>
                    {editing ? null : (
                      <span className="flex min-w-0 items-center gap-2 text-xs text-muted-foreground">
                        <Badge tone={protocol === 'https' ? 'success' : 'neutral'} size="sm">
                          {protocol.toUpperCase()}
                        </Badge>
                        <span className="truncate font-mono">{address}</span>
                        <span className="shrink-0 tabular">Weight {weight}</span>
                      </span>
                    )}
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    <Button
                      type="button"
                      variant="ghost"
                      size="xs"
                      aria-expanded={editing}
                      onClick={() => setEditingIndex(editing ? null : index)}
                    >
                      {editing ? 'Done' : 'Edit'}
                      <ChevronDown
                        className={cn('transition-transform duration-200', editing && 'rotate-180')}
                      />
                    </Button>
                    <Button
                      type="button"
                      variant="destructive-ghost"
                      size="xs"
                      onClick={() => handleDelete(index)}
                    >
                      Delete
                    </Button>
                  </div>
                </div>
                {editing ? (
                  <div className="grid grid-cols-1 gap-3 border-t border-border px-3.5 py-3.5 sm:grid-cols-2">
                    <div className="flex flex-col gap-1.5">
                      <label className="text-xs font-medium" htmlFor={fieldId('name')}>
                        Name
                      </label>
                      <Input
                        id={fieldId('name')}
                        className="h-8 font-mono text-xs"
                        value={name}
                        onChange={(e) => handleUpdate(index, 'name', e.target.value)}
                        placeholder="endpoint-1"
                      />
                    </div>
                    <div className="flex flex-col gap-1.5">
                      <label className="text-xs font-medium" htmlFor={fieldId('address')}>
                        Address
                      </label>
                      <Input
                        id={fieldId('address')}
                        className="h-8 font-mono text-xs"
                        value={address}
                        onChange={(e) => handleUpdate(index, 'endpoint', e.target.value)}
                        placeholder="localhost:8000"
                      />
                    </div>
                    <div className="flex flex-col gap-1.5">
                      <label className="text-xs font-medium" htmlFor={fieldId('protocol')}>
                        Protocol
                      </label>
                      <NativeSelect
                        id={fieldId('protocol')}
                        size="sm"
                        value={protocol}
                        onChange={(e) => handleUpdate(index, 'protocol', e.target.value)}
                      >
                        <option value="http">HTTP</option>
                        <option value="https">HTTPS</option>
                      </NativeSelect>
                    </div>
                    <div className="flex flex-col gap-1.5">
                      <label className="text-xs font-medium" htmlFor={fieldId('weight')}>
                        Weight
                      </label>
                      <Input
                        id={fieldId('weight')}
                        className="h-8 tabular"
                        type="number"
                        min="1"
                        value={weight}
                        onChange={(e) =>
                          handleUpdate(index, 'weight', parseInt(e.target.value) || 1)
                        }
                        placeholder="1"
                      />
                    </div>
                  </div>
                ) : null}
              </div>
            )
          })}
        </div>
      ) : (
        <p className="rounded-lg border border-dashed border-border-strong px-3 py-4 text-center text-xs text-muted-foreground">
          No endpoints configured. Add one to route traffic to this model.
        </p>
      )}

      <Button type="button" variant="outline" size="xs" className="self-start" onClick={handleAdd}>
        <Plus />
        Add Endpoint
      </Button>
    </div>
  )
}

export default EndpointsEditor
