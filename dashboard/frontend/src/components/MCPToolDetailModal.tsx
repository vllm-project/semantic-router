import React from 'react'
import { X } from 'lucide-react'

import {
  Badge,
  Button,
  Dialog,
  DialogBody,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogTitle,
  IconTile,
} from '../ui'
import type { UnifiedTool } from './mcpConfigPanelTypes'
import { TOOL_SOURCE_META, toolIcon } from './mcpConfigPanelUtils'

interface MCPToolDetailModalProps {
  tool: UnifiedTool
  onClose: () => void
}

export const MCPToolDetailModal: React.FC<MCPToolDetailModalProps> = ({ tool, onClose }) => {
  const source = TOOL_SOURCE_META[tool.sourceType]
  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent size="lg" hideClose>
        <div className="flex items-start gap-3 px-6 pb-2 pt-5">
          <IconTile icon={toolIcon(tool.name)} tone="slate" />
          <div className="flex min-w-0 flex-1 flex-col gap-1">
            <div className="flex flex-wrap items-center gap-2">
              <DialogTitle className="truncate font-mono text-[15px]">{tool.name}</DialogTitle>
              <Badge tone={source.tone}>{source.label}</Badge>
            </div>
            <DialogDescription>{tool.description || 'No description'}</DialogDescription>
          </div>
          <DialogClose asChild>
            <Button variant="ghost" size="icon-sm" aria-label="Close tool details">
              <X />
            </Button>
          </DialogClose>
        </div>
        <DialogBody className="flex flex-col gap-4 pb-5">
          <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-[13px]">
            <div className="flex flex-col gap-1">
              <dt className="text-xs text-subtle-foreground">Source</dt>
              <dd className="font-mono text-[12.5px]">{tool.source}</dd>
            </div>
            <div className="flex flex-col gap-1">
              <dt className="text-xs text-subtle-foreground">Parameters</dt>
              <dd className="tabular">{tool.parameters.length}</dd>
            </div>
          </dl>
          {tool.parameters.length === 0 ? (
            <p className="rounded-lg border border-dashed border-border px-3 py-3 text-[13px] text-muted-foreground">
              This tool requires no parameters.
            </p>
          ) : (
            <div className="overflow-hidden rounded-lg border border-border">
              {tool.parameters.map((parameter) => (
                <div
                  key={parameter.name}
                  className="flex flex-col gap-1 border-b border-border px-3.5 py-2.5 last:border-0"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <code className="font-mono text-[12.5px] font-medium">{parameter.name}</code>
                    <Badge size="sm" tone="outline" className="font-mono">
                      {parameter.type}
                    </Badge>
                    {parameter.required ? (
                      <Badge size="sm" tone="warning">
                        Required
                      </Badge>
                    ) : null}
                  </div>
                  {parameter.description ? (
                    <p className="text-xs leading-relaxed text-muted-foreground">
                      {parameter.description}
                    </p>
                  ) : null}
                </div>
              ))}
            </div>
          )}
        </DialogBody>
        <DialogFooter>
          <Button variant="ghost" onClick={onClose}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
