// CustomNodes/PluginChainNode.tsx - Plugin chain node with collapse support

import React, { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { PluginConfig } from '../../types'
import { PLUGIN_ICONS, PLUGIN_COLORS } from '../../constants'
import styles from './CustomNodes.module.css'

interface PluginChainNodeData {
  decisionName: string
  plugins: PluginConfig[]
  collapsed?: boolean
  isHighlighted?: boolean
  onToggleCollapse?: () => void
  // Global config for showing inheritance
  globalCacheEnabled?: boolean
  globalCacheThreshold?: number
}

// Check if plugin overrides global config
function getPluginOverrideInfo(plugin: PluginConfig, data: PluginChainNodeData): string | null {
  if (plugin.type === 'semantic-cache') {
    const config = plugin.configuration as { similarity_threshold?: number; enabled?: boolean } | undefined
    if (config?.enabled === false) {
      return '(disabled)'
    }
    if (config?.similarity_threshold && data.globalCacheThreshold) {
      if (config.similarity_threshold !== data.globalCacheThreshold) {
        return `(${config.similarity_threshold})`
      }
    }
    if (data.globalCacheEnabled) {
      return '(global)'
    }
  }
  return null
}

export const PluginChainNode = memo<NodeProps<PluginChainNodeData>>(({ data }) => {
  const { plugins, collapsed = false, isHighlighted, onToggleCollapse } = data

  return (
    <div
      className={`${styles.pluginChainNode} ${isHighlighted ? styles.highlighted : ''}`}
    >
      <Handle type="target" position={Position.Top} />

      <div
        className={styles.pluginChainHeader}
        onClick={onToggleCollapse}
      >
        <span className={styles.collapseIcon}>{collapsed ? '▶' : '▼'}</span>
        <span className={styles.pluginChainTitle}>
          🔌 Plugin Chain ({plugins.length})
        </span>
      </div>

      {!collapsed && (
        <div className={styles.pluginChain}>
          {plugins.map((plugin, idx) => {
            const colors = PLUGIN_COLORS[plugin.type] || { background: '#607D8B', border: '#455A64' }
            const icon = PLUGIN_ICONS[plugin.type] || '🔌'
            const overrideInfo = getPluginOverrideInfo(plugin, data)

            return (
              <React.Fragment key={plugin.type}>
                <div
                  className={`${styles.chainPlugin} ${!plugin.enabled ? styles.disabled : ''}`}
                  style={{ background: colors.background }}
                  title={plugin.enabled ? plugin.type : `${plugin.type} (disabled)`}
                >
                  <span>{icon}</span>
                  <span>
                    {plugin.type.replace(/[-_]/g, ' ')}
                    {overrideInfo && (
                      <span className={styles.pluginOverride}>{overrideInfo}</span>
                    )}
                  </span>
                </div>
                {idx < plugins.length - 1 && (
                  <span className={styles.chainArrow}>↓</span>
                )}
              </React.Fragment>
            )
          })}
        </div>
      )}

      <Handle type="source" position={Position.Bottom} />
    </div>
  )
})

PluginChainNode.displayName = 'PluginChainNode'
