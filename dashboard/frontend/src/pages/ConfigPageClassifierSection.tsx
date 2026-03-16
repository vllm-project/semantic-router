import styles from './ConfigPage.module.css'
import type { MCPCategoryModel, ModelConfig, PreferenceModelConfig } from './configPageSupport'
import { formatThreshold } from './configPageSupport'
import { cloneConfig, type RouterSectionBaseProps } from './configPageRouterSectionSupport'

export default function ConfigPageClassifierSection({
  config,
  routerConfig,
  isReadonly,
  openEditModal,
  saveConfig,
}: RouterSectionBaseProps) {
  const renderClassifyBERT = () => {
    const categoryModel = routerConfig.classifier?.category_model
    const mcpCategoryModel = routerConfig.classifier?.mcp_category_model
    const hasInTree = Boolean(categoryModel)
    const hasOutTree = Boolean(mcpCategoryModel?.enabled)

    return (
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <h3 className={styles.sectionTitle}>Classify BERT Model</h3>
        </div>
        <div className={styles.sectionContent}>
          {hasInTree && categoryModel && (
            <div className={styles.modelCard}>
              <div className={styles.modelCardHeader}>
                <span className={styles.modelCardTitle}>In-tree Category Classifier</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <span className={`${styles.statusBadge} ${styles.statusActive}`}>
	                    {categoryModel.use_cpu ? 'CPU' : 'GPU'}
                  </span>
                  {!isReadonly && (
                    <button
                      className={styles.editButton}
	                      onClick={() => {
	                        openEditModal<ModelConfig>(
	                          'Edit In-tree Category Classifier',
	                          categoryModel,
                          [
                            { name: 'model_id', label: 'Model ID', type: 'text', required: true, placeholder: 'e.g., answerdotai/ModernBERT-base', description: 'HuggingFace model ID for category classification' },
                            { name: 'threshold', label: 'Classification Threshold', type: 'percentage', required: true, placeholder: '70', description: 'Confidence threshold for category classification (0-100%)', step: 1 },
                            { name: 'use_cpu', label: 'Use CPU', type: 'boolean', description: 'Use CPU instead of GPU for inference' },
                            { name: 'use_modernbert', label: 'Use ModernBERT', type: 'boolean', description: 'Enable ModernBERT-based classification' },
                            { name: 'category_mapping_path', label: 'Category Mapping Path', type: 'text', placeholder: 'config/category_mapping.json', description: 'Path to category mapping configuration' },
                          ],
                          async (data) => {
                            const newConfig = cloneConfig(config)
                            if (!newConfig.classifier) newConfig.classifier = {}
	                            newConfig.classifier.category_model = data
	                            await saveConfig(newConfig)
	                          }
	                        )
                      }}
                    />
                  )}
                </div>
              </div>
              <div className={styles.modelCardBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Type</span>
                  <span className={`${styles.badge} ${styles.badgeInfo}`}>Built-in ModernBERT</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Model ID</span>
	                  <span className={styles.configValue}>{categoryModel.model_id}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Threshold</span>
	                  <span className={styles.configValue}>{formatThreshold(categoryModel.threshold)}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>ModernBERT</span>
	                  <span className={`${styles.statusBadge} ${categoryModel.use_modernbert ? styles.statusActive : styles.statusInactive}`}>
	                    {categoryModel.use_modernbert ? '✓ Enabled' : '✗ Disabled'}
	                  </span>
	                </div>
	                {categoryModel.category_mapping_path && (
	                  <div className={styles.configRow}>
	                    <span className={styles.configLabel}>Mapping Path</span>
	                    <span className={styles.configValue}>{categoryModel.category_mapping_path}</span>
	                  </div>
	                )}
              </div>
            </div>
          )}

          {hasOutTree && mcpCategoryModel && (
            <div className={styles.modelCard}>
              <div className={styles.modelCardHeader}>
                <span className={styles.modelCardTitle}>Out-tree Category Classifier (MCP)</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <span className={`${styles.statusBadge} ${styles.statusActive}`}>✓ Enabled</span>
                  {!isReadonly && (
                    <button
                      className={styles.editButton}
	                      onClick={() => {
	                        openEditModal<MCPCategoryModel>(
	                          'Edit Out-tree MCP Category Classifier',
	                          mcpCategoryModel,
                          [
                            { name: 'enabled', label: 'Enable MCP Classifier', type: 'boolean', description: 'Enable or disable MCP-based classification' },
                            { name: 'transport_type', label: 'Transport Type', type: 'select', options: ['stdio', 'http'], required: true, description: 'MCP transport protocol type' },
                            { name: 'command', label: 'Command', type: 'text', placeholder: 'e.g., python mcp_server.py', description: 'Command to start MCP server (for stdio transport)' },
                            { name: 'args', label: 'Arguments (JSON)', type: 'json', placeholder: '["--port", "8080"]', description: 'Command line arguments as JSON array' },
                            { name: 'env', label: 'Environment Variables (JSON)', type: 'json', placeholder: '{"API_KEY": "xxx"}', description: 'Environment variables as JSON object' },
                            { name: 'url', label: 'URL', type: 'text', placeholder: 'http://localhost:8080', description: 'MCP server URL (for http transport)' },
                            { name: 'tool_name', label: 'Tool Name', type: 'text', placeholder: 'classify_category', description: 'Name of the MCP tool to call' },
                            { name: 'threshold', label: 'Classification Threshold', type: 'percentage', required: true, placeholder: '70', description: 'Confidence threshold for classification (0-100%)', step: 1 },
                            { name: 'timeout_seconds', label: 'Timeout (seconds)', type: 'number', placeholder: '30', description: 'Request timeout in seconds' },
                          ],
                          async (data) => {
                            const newConfig = cloneConfig(config)
                            if (!newConfig.classifier) newConfig.classifier = {}
                            newConfig.classifier.mcp_category_model = data
                            await saveConfig(newConfig)
                          }
                        )
                      }}
                    />
                  )}
                </div>
              </div>
              <div className={styles.modelCardBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Type</span>
                  <span className={`${styles.badge} ${styles.badgeInfo}`}>MCP Protocol</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Transport Type</span>
	                  <span className={styles.configValue}>{mcpCategoryModel.transport_type}</span>
	                </div>
	                {mcpCategoryModel.command && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Command</span>
	                    <span className={styles.configValue}>{mcpCategoryModel.command}</span>
	                  </div>
	                )}
	                {mcpCategoryModel.url && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>URL</span>
	                    <span className={styles.configValue}>{mcpCategoryModel.url}</span>
	                  </div>
	                )}
	                {mcpCategoryModel.tool_name && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Tool Name</span>
	                    <span className={styles.configValue}>{mcpCategoryModel.tool_name}</span>
	                  </div>
	                )}
	                <div className={styles.configRow}>
	                  <span className={styles.configLabel}>Threshold</span>
	                  <span className={styles.configValue}>{formatThreshold(mcpCategoryModel.threshold)}</span>
	                </div>
	                {mcpCategoryModel.timeout_seconds && (
	                  <div className={styles.configRow}>
	                    <span className={styles.configLabel}>Timeout</span>
	                    <span className={styles.configValue}>{mcpCategoryModel.timeout_seconds}s</span>
	                  </div>
	                )}
              </div>
            </div>
          )}

          {!hasInTree && !hasOutTree && (
            <div className={styles.emptyState}>No category classifier configured</div>
          )}
        </div>
      </div>
    )
  }

  const renderPreferenceModel = () => {
    const preferenceModel = routerConfig.classifier?.preference_model
    const isContrastive = Boolean(preferenceModel?.use_contrastive)
    const preferenceModelForm: PreferenceModelConfig = preferenceModel || { use_contrastive: true, embedding_model: '' }

    return (
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <h3 className={styles.sectionTitle}>Preference Model</h3>
          <button
            hidden
            className={styles.sectionEditButton}
	            onClick={() => {
	              openEditModal<PreferenceModelConfig>(
	                preferenceModel ? 'Edit Preference Model' : 'Add Preference Model',
	                preferenceModelForm,
                [
                  { name: 'use_contrastive', label: 'Use Contrastive Preference', type: 'boolean', description: 'Enable embedding-based contrastive routing instead of external LLM routing' },
                  { name: 'embedding_model', label: 'Embedding Model', type: 'select', options: ['', 'mmbert', 'qwen3', 'gemma'], placeholder: 'Select embedding model', description: 'Embedding backbone for contrastive preference routing (requires embedding_models path)', shouldHide: (data) => !data.use_contrastive },
                ],
                async (data) => {
                  const newConfig = cloneConfig(config)
                  if (!newConfig.classifier) newConfig.classifier = {}
                  newConfig.classifier.preference_model = data
                  await saveConfig(newConfig)
                },
                preferenceModel ? 'edit' : 'add'
              )
            }}
          >
            {preferenceModel ? 'Edit' : 'Add'}
          </button>
        </div>
        <div className={styles.sectionContent}>
          {preferenceModel ? (
            <div className={styles.modelCard}>
              <div className={styles.modelCardHeader}>
                <span className={styles.modelCardTitle}>{isContrastive ? 'Contrastive Preference Classifier' : 'External Preference Classifier'}</span>
                <span className={`${styles.statusBadge} ${styles.statusActive}`}>{isContrastive ? 'Contrastive' : 'External'}</span>
              </div>
              <div className={styles.modelCardBody}>
                {isContrastive ? (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Embedding Model</span>
                    <span className={styles.configValue}>{preferenceModel.embedding_model || 'Not set'}</span>
                  </div>
                ) : (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Mode</span>
                    <span className={styles.configValue}>Uses external LLM preference classifier</span>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className={styles.emptyState}>Preference model not configured</div>
          )}
        </div>
      </div>
    )
  }

  return (
    <>
      {renderClassifyBERT()}
      {renderPreferenceModel()}
    </>
  )
}
