// CustomNodes/GlobalPluginNode.tsx - Global plugin node (Jailbreak, PII, Cache)

import { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { GlobalPluginConfig } from '../../types'
import { GLOBAL_PLUGIN_DISPLAY } from '../../constants'
import styles from './CustomNodes.module.css'

interface GlobalPluginNodeData {
  plugin: GlobalPluginConfig
  isHighlighted?: boolean
}

// Get status text based on plugin type and mode
function getPluginStatus(plugin: GlobalPluginConfig): { text: string; tooltip: string } {
  if (!plugin.enabled) {
    return { text: '✗ Disabled', tooltip: 'Plugin is disabled' }
  }

  // PII Detection: Model loaded but requires decision-level activation
  if (plugin.type === 'pii_detection') {
    const mode = plugin.config?.mode as string
    if (mode === 'model_loaded') {
      return {
        text: '◐ Model Loaded',
        tooltip: 'Model loaded. Detection requires pii plugin in decision.',
      }
    }
  }

  // Jailbreak: Active for all requests (global AND decision)
  if (plugin.type === 'prompt_guard') {
    return {
      text: `✓ Active`,
      tooltip: `Threshold: ${plugin.threshold || 0.7}. Can be overridden per-decision.`,
    }
  }

  // Semantic Cache: Active globally, can be overridden
  if (plugin.type === 'semantic_cache') {
    const threshold = plugin.config?.similarity_threshold as number
    return {
      text: `✓ Active`,
      tooltip: `Threshold: ${threshold || 0.85}. Can be overridden per-decision.`,
    }
  }

  return { text: `✓ ${plugin.modelId || 'Enabled'}`, tooltip: '' }
}

export const GlobalPluginNode = memo<NodeProps<GlobalPluginNodeData>>(({ data }) => {
  const { plugin, isHighlighted } = data
  const display = GLOBAL_PLUGIN_DISPLAY[plugin.type] || {
    icon: '🔌',
    label: plugin.type,
    color: '#607D8B',
  }

  const status = getPluginStatus(plugin)
  const isPartiallyActive = plugin.type === 'pii_detection' && plugin.config?.mode === 'model_loaded'

  return (
    <div
      className={`${styles.globalPluginNode} ${!plugin.enabled ? styles.disabled : ''} ${isHighlighted ? styles.highlighted : ''}`}
      style={{
        background: plugin.enabled ? display.color : undefined,
        borderColor: plugin.enabled ? display.color : undefined,
        border: `2px solid ${plugin.enabled ? display.color : '#616161'}`,
        opacity: isPartiallyActive ? 0.85 : 1,
      }}
      title={status.tooltip}
    >
      <Handle type="target" position={Position.Top} />

      <div className={styles.pluginHeader}>
        <span className={styles.pluginIcon}>{display.icon}</span>
        <span className={styles.pluginTitle}>{display.label}</span>
      </div>

      <div className={styles.pluginStatus}>
        {status.text}
      </div>

      <Handle type="source" position={Position.Bottom} />
    </div>
  )
})

GlobalPluginNode.displayName = 'GlobalPluginNode'
