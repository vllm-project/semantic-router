// CustomNodes/PluginChainNode.tsx - Plugin chain node with collapse support

import React, { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { PluginConfig, AlgorithmConfig } from '../../types'
import { PLUGIN_COLORS, ALGORITHM_COLORS } from '../../constants'
import ProductIcon from '../../../../components/ProductIcon'
import styles from './CustomNodes.module.css'

interface PluginChainNodeData {
  decisionName: string
  plugins: PluginConfig[]
  algorithm?: AlgorithmConfig // Optional algorithm to display at the top
  collapsed?: boolean
  isHighlighted?: boolean
  onToggleCollapse?: () => void
  // Global config for showing inheritance
  globalCacheEnabled?: boolean
  globalCacheThreshold?: number
}

// Check if plugin overrides global config
function getPluginOverrideInfo(plugin: PluginConfig, data: PluginChainNodeData): string | null {
  if (plugin.type === 'response_cache') {
    const config = plugin.configuration as
      | { similarity_threshold?: number; enabled?: boolean }
      | undefined
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
  const { plugins, algorithm, collapsed = false, isHighlighted, onToggleCollapse } = data

  // Calculate total items (algorithm + plugins)
  const totalItems = (algorithm ? 1 : 0) + plugins.length

  return (
    <div className={`${styles.pluginChainNode} ${isHighlighted ? styles.highlighted : ''}`}>
      <Handle type="target" position={Position.Left} />

      <div className={styles.pluginChainHeader} onClick={onToggleCollapse}>
        <ProductIcon
          className={`${styles.collapseIcon} ${collapsed ? '' : styles.collapseIconOpen}`}
          name="chevron-right"
          aria-hidden="true"
        />
        <span className={styles.pluginChainTitle}>
          <ProductIcon name="plug" aria-hidden="true" />
          Plugin Chain ({totalItems})
        </span>
      </div>

      {!collapsed && (
        <div className={styles.pluginChain}>
          {/* Display algorithm first if present */}
          {algorithm && (
            <>
              <div
                className={styles.chainPlugin}
                style={{ background: ALGORITHM_COLORS[algorithm.type]?.background || '#8f949c' }}
                title={algorithm.type}
              >
                <ProductIcon name="settings" aria-hidden="true" />
                <span>{algorithm.type}</span>
              </div>
              {plugins.length > 0 && (
                <ProductIcon
                  className={styles.chainArrow}
                  name="chevron-right"
                  aria-hidden="true"
                />
              )}
            </>
          )}

          {/* Display plugins */}
          {plugins.map((plugin, idx) => {
            const colors = PLUGIN_COLORS[plugin.type] || {
              background: '#607D8B',
              border: '#455A64',
            }
            const overrideInfo = getPluginOverrideInfo(plugin, data)

            return (
              <React.Fragment key={plugin.type}>
                <div
                  className={`${styles.chainPlugin} ${!plugin.enabled ? styles.disabled : ''}`}
                  style={{ background: colors.background }}
                  title={plugin.enabled ? plugin.type : `${plugin.type} (disabled)`}
                >
                  <ProductIcon name="plug" aria-hidden="true" />
                  <span>
                    {plugin.type.replace(/[-_]/g, ' ')}
                    {overrideInfo && <span className={styles.pluginOverride}>{overrideInfo}</span>}
                  </span>
                </div>
                {idx < plugins.length - 1 && (
                  <ProductIcon
                    className={styles.chainArrow}
                    name="chevron-right"
                    aria-hidden="true"
                  />
                )}
              </React.Fragment>
            )
          })}
        </div>
      )}

      <Handle type="source" position={Position.Right} />
    </div>
  )
})

PluginChainNode.displayName = 'PluginChainNode'
