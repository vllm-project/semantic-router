import styles from './ConfigPage.module.css'
import type { EmbeddingModelsConfig, ModelConfig } from './configPageSupport'
import { formatThreshold } from './configPageSupport'
import { cloneConfig, type RouterSectionBaseProps } from './configPageRouterSectionSupport'

export default function ConfigPageSafetyCacheSection({
  config,
  routerConfig,
  isReadonly,
  openEditModal,
  saveConfig,
}: RouterSectionBaseProps) {
  const piiModel = routerConfig.classifier?.pii_model
  const promptGuard = routerConfig.prompt_guard
  const embeddingModels = routerConfig.embedding_models
  const semanticCache = routerConfig.semantic_cache

  const renderPIIModernBERT = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <h3 className={styles.sectionTitle}>PII Detection (ModernBERT)</h3>
        {piiModel && !isReadonly && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
	              openEditModal<ModelConfig>(
	                'Edit PII Detection Configuration',
	                piiModel,
                [
                  { name: 'model_id', label: 'Model ID', type: 'text', required: true, placeholder: 'e.g., answerdotai/ModernBERT-base', description: 'HuggingFace model ID for PII detection' },
                  { name: 'threshold', label: 'Detection Threshold', type: 'percentage', required: true, placeholder: '50', description: 'Confidence threshold for PII detection (0-100%)', step: 1 },
                  { name: 'use_cpu', label: 'Use CPU', type: 'boolean', description: 'Use CPU instead of GPU for inference' },
                  { name: 'use_modernbert', label: 'Use ModernBERT', type: 'boolean', description: 'Enable ModernBERT-based PII detection' },
                  { name: 'pii_mapping_path', label: 'PII Mapping Path', type: 'text', placeholder: 'config/pii_mapping.json', description: 'Path to PII entity mapping configuration' },
                ],
                async (data) => {
                  const newConfig = cloneConfig(config)
                  if (!newConfig.classifier) newConfig.classifier = {}
                  newConfig.classifier.pii_model = data
                  await saveConfig(newConfig)
                }
              )
            }}
          >
            Edit
          </button>
        )}
      </div>
      <div className={styles.sectionContent}>
	        {piiModel ? (
          <div className={styles.modelCard}>
            <div className={styles.modelCardHeader}>
              <span className={styles.modelCardTitle}>PII Classifier Model</span>
              <span className={`${styles.statusBadge} ${styles.statusActive}`}>
	                {piiModel.use_cpu ? 'CPU' : 'GPU'}
              </span>
            </div>
            <div className={styles.modelCardBody}>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Model ID</span>
	                <span className={styles.configValue}>{piiModel.model_id}</span>
              </div>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Threshold</span>
	                <span className={styles.configValue}>{formatThreshold(piiModel.threshold)}</span>
              </div>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>ModernBERT</span>
	                <span className={`${styles.statusBadge} ${piiModel.use_modernbert ? styles.statusActive : styles.statusInactive}`}>
	                  {piiModel.use_modernbert ? '✓ Enabled' : '✗ Disabled'}
	                </span>
	              </div>
	              {piiModel.pii_mapping_path && (
	                <div className={styles.configRow}>
	                  <span className={styles.configLabel}>Mapping Path</span>
	                  <span className={styles.configValue}>{piiModel.pii_mapping_path}</span>
	                </div>
	              )}
            </div>
          </div>
        ) : (
          <div className={styles.emptyState}>PII detection not configured</div>
        )}
      </div>
    </div>
  )

  const renderJailbreakModernBERT = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <h3 className={styles.sectionTitle}>Jailbreak Detection (ModernBERT)</h3>
        {promptGuard && !isReadonly && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
	              openEditModal<ModelConfig & { enabled: boolean }>(
	                'Edit Jailbreak Detection Configuration',
	                promptGuard,
                [
                  { name: 'enabled', label: 'Enable Jailbreak Detection', type: 'boolean', description: 'Enable or disable jailbreak detection' },
                  { name: 'model_id', label: 'Model ID', type: 'text', required: true, placeholder: 'e.g., answerdotai/ModernBERT-base', description: 'HuggingFace model ID for jailbreak detection' },
                  { name: 'threshold', label: 'Detection Threshold', type: 'percentage', required: true, placeholder: '50', description: 'Confidence threshold for jailbreak detection (0-100%)', step: 1 },
                  { name: 'use_cpu', label: 'Use CPU', type: 'boolean', description: 'Use CPU instead of GPU for inference' },
                  { name: 'use_modernbert', label: 'Use ModernBERT', type: 'boolean', description: 'Enable ModernBERT-based jailbreak detection' },
                  { name: 'jailbreak_mapping_path', label: 'Jailbreak Mapping Path', type: 'text', placeholder: 'config/jailbreak_mapping.json', description: 'Path to jailbreak pattern mapping configuration' },
                ],
                async (data) => {
                  const newConfig = cloneConfig(config)
                  newConfig.prompt_guard = data
                  await saveConfig(newConfig)
                }
              )
            }}
          >
            Edit
          </button>
        )}
      </div>
      <div className={styles.sectionContent}>
	        {promptGuard ? (
          <div className={styles.modelCard}>
            <div className={styles.modelCardHeader}>
              <span className={styles.modelCardTitle}>Jailbreak Protection</span>
	              <span className={`${styles.statusBadge} ${promptGuard.enabled ? styles.statusActive : styles.statusInactive}`}>
	                {promptGuard.enabled ? '✓ Enabled' : '✗ Disabled'}
	              </span>
	            </div>
	            {promptGuard.enabled && (
	              <div className={styles.modelCardBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Model ID</span>
	                  <span className={styles.configValue}>{promptGuard.model_id}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Threshold</span>
	                  <span className={styles.configValue}>{formatThreshold(promptGuard.threshold)}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Use CPU</span>
                  <span className={`${styles.statusBadge} ${styles.statusActive}`}>
	                    {promptGuard.use_cpu ? 'CPU' : 'GPU'}
                  </span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>ModernBERT</span>
	                  <span className={`${styles.statusBadge} ${promptGuard.use_modernbert ? styles.statusActive : styles.statusInactive}`}>
	                    {promptGuard.use_modernbert ? '✓ Enabled' : '✗ Disabled'}
	                  </span>
	                </div>
	                {promptGuard.jailbreak_mapping_path && (
	                  <div className={styles.configRow}>
	                    <span className={styles.configLabel}>Mapping Path</span>
	                    <span className={styles.configValue}>{promptGuard.jailbreak_mapping_path}</span>
	                  </div>
	                )}
              </div>
            )}
          </div>
        ) : (
          <div className={styles.emptyState}>Jailbreak detection not configured</div>
        )}
      </div>
    </div>
  )

  const renderSimilarityBERT = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <h3 className={styles.sectionTitle}>Similarity Embedding Configuration</h3>
        {embeddingModels && !isReadonly && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
	              openEditModal<ModelConfig>(
	                'Edit Similarity Embedding Configuration',
	                {
                    model_id: embeddingModels.bert_model_path || '',
                    threshold: embeddingModels.embedding_config?.min_score_threshold || 0.5,
                    use_cpu: embeddingModels.use_cpu ?? true,
                  },
                [
                  { name: 'model_id', label: 'Model ID', type: 'text', required: true, placeholder: 'e.g., sentence-transformers/all-MiniLM-L6-v2', description: 'HuggingFace model ID for semantic similarity' },
                  { name: 'threshold', label: 'Similarity Threshold', type: 'percentage', required: true, placeholder: '80', description: 'Minimum similarity score for cache hits (0-100%)', step: 1 },
                  { name: 'use_cpu', label: 'Use CPU', type: 'boolean', description: 'Use CPU instead of GPU for inference' },
                ],
                async (data) => {
                  const newConfig = cloneConfig(config)
                  const nextEmbeddingModels: EmbeddingModelsConfig = {
                    ...(newConfig.embedding_models || {}),
                    bert_model_path: data.model_id,
                    use_cpu: Boolean(data.use_cpu),
                    embedding_config: {
                      ...(newConfig.embedding_models?.embedding_config || {}),
                      min_score_threshold: data.threshold,
                    },
                  }
                  newConfig.embedding_models = nextEmbeddingModels
                  await saveConfig(newConfig)
                }
              )
            }}
          >
            Edit
          </button>
        )}
      </div>
      <div className={styles.sectionContent}>
	        {embeddingModels?.bert_model_path ? (
          <div className={styles.modelCard}>
            <div className={styles.modelCardHeader}>
              <span className={styles.modelCardTitle}>Embedding Model (Semantic Similarity)</span>
              <span className={`${styles.statusBadge} ${styles.statusActive}`}>
	                {embeddingModels.use_cpu ? 'CPU' : 'GPU'}
              </span>
            </div>
            <div className={styles.modelCardBody}>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Model ID</span>
	                <span className={styles.configValue}>{embeddingModels.bert_model_path}</span>
              </div>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Threshold</span>
	                <span className={styles.configValue}>{formatThreshold(embeddingModels.embedding_config?.min_score_threshold || 0.5)}</span>
              </div>
            </div>
          </div>
        ) : (
          <div className={styles.emptyState}>Similarity embedding model not configured</div>
        )}

	        {semanticCache && (
          <div className={styles.featureCard}>
            <div className={styles.featureHeader}>
              <span className={styles.featureTitle}>Semantic Cache</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
	                <span className={`${styles.statusBadge} ${semanticCache.enabled ? styles.statusActive : styles.statusInactive}`}>
	                  {semanticCache.enabled ? '✓ Enabled' : '✗ Disabled'}
	                </span>
                {!isReadonly && (
                  <button
                    className={styles.sectionEditButton}
                    onClick={() => {
                      openEditModal(
                        'Edit Semantic Cache Configuration',
                        config?.semantic_cache || {},
                        [
                          { name: 'enabled', label: 'Enable Semantic Cache', type: 'boolean', description: 'Enable or disable semantic caching' },
                          { name: 'backend_type', label: 'Backend Type', type: 'select', options: ['memory', 'redis', 'memcached'], description: 'Cache backend storage type' },
                          { name: 'similarity_threshold', label: 'Similarity Threshold', type: 'percentage', required: true, placeholder: '90', description: 'Minimum similarity score for cache hits (0-100%)', step: 1 },
                          { name: 'max_entries', label: 'Max Entries', type: 'number', placeholder: '10000', description: 'Maximum number of cached entries' },
                          { name: 'ttl_seconds', label: 'TTL (seconds)', type: 'number', placeholder: '3600', description: 'Time-to-live for cached entries' },
                          { name: 'eviction_policy', label: 'Eviction Policy', type: 'select', options: ['lru', 'lfu', 'fifo'], description: 'Cache eviction policy when max entries reached' },
                        ],
                        async (data) => {
                          const newConfig = cloneConfig(config)
                          newConfig.semantic_cache = data
                          await saveConfig(newConfig)
                        }
                      )
                    }}
                  >
                    Edit
                  </button>
                )}
              </div>
            </div>
	            {semanticCache.enabled && (
	              <div className={styles.featureBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Backend Type</span>
	                  <span className={styles.configValue}>{semanticCache.backend_type || 'memory'}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Similarity Threshold</span>
	                  <span className={styles.configValue}>{formatThreshold(semanticCache.similarity_threshold ?? 0)}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Max Entries</span>
	                  <span className={styles.configValue}>{semanticCache.max_entries}</span>
                </div>
	                <div className={styles.configRow}>
	                  <span className={styles.configLabel}>TTL</span>
	                  <span className={styles.configValue}>{semanticCache.ttl_seconds}s</span>
	                </div>
	                {semanticCache.eviction_policy && (
	                  <div className={styles.configRow}>
	                    <span className={styles.configLabel}>Eviction Policy</span>
	                    <span className={styles.configValue}>{semanticCache.eviction_policy}</span>
	                  </div>
	                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )

  return (
    <>
      {renderSimilarityBERT()}
      {renderPIIModernBERT()}
      {renderJailbreakModernBERT()}
    </>
  )
}
