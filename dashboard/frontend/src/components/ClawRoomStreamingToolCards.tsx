import { useMemo, useState } from 'react'

import { ToolCard } from './ChatComponentToolCards'
import styles from './ClawRoomChat.module.css'
import {
  toPlaygroundToolCall,
  toPlaygroundToolResult,
  type ClawRoomToolTraceStep,
} from './clawRoomToolTrace'

interface ClawRoomStreamingToolCardsProps {
  steps: ClawRoomToolTraceStep[]
}

const ClawRoomStreamingToolCards = ({ steps }: ClawRoomStreamingToolCardsProps) => {
  const [expandedToolCards, setExpandedToolCards] = useState<Set<string>>(new Set())

  const toolModels = useMemo(
    () => steps.map(step => ({
      step,
      toolCall: toPlaygroundToolCall(step),
      toolResult: toPlaygroundToolResult(step),
    })),
    [steps]
  )

  if (toolModels.length === 0) {
    return null
  }

  return (
    <div className={styles.toolTraceList} data-testid="claw-room-tool-trace">
      {toolModels.map(({ step, toolCall, toolResult }) => (
        <ToolCard
          key={step.id}
          toolCall={toolCall}
          toolResult={toolResult}
          isExpanded={expandedToolCards.has(step.id)}
          onToggle={() => {
            setExpandedToolCards(previous => {
              const next = new Set(previous)
              if (next.has(step.id)) {
                next.delete(step.id)
              } else {
                next.add(step.id)
              }
              return next
            })
          }}
        />
      ))}
    </div>
  )
}

export default ClawRoomStreamingToolCards
