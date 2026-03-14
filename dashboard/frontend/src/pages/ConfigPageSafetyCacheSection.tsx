import styles from './ConfigPage.module.css'
import { formatThreshold } from './configPageSupport'
import { cloneConfig, type RouterSectionBaseProps } from './configPageRouterSectionSupport'

export default function ConfigPageSafetyCacheSection({
  config,
  routerConfig,
  isReadonly,
  openEditModal,
  saveConfig,
}: RouterSectionBaseProps) {
  const renderPIIModernBERT = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <h3 className={styles.sectionTitle}>PII Detection (ModernBERT)</h3>
        {routerConfig.classifier?.pii_model && !isReadonly && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
              openEditModal(
                'Edit PII Detection Configuration',
                routerConfig.classifier?.pii_model || {},
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
        {routerConfig.classifier?.pii_model ? (
          <div className={styles.modelCard}>
            <div className={styles.modelCardHeader}>
              <span className={styles.modelCardTitle}>PII Classifier Model</span>
              <span className={`${styles.statusBadge} ${styles.statusActive}`}>
                {routerConfig.classifier.pii_model.use_cpu ? 'CPU' : 'GPU'}
              </span>
            </div>
            <div className={styles.modelCardBody}>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Model ID</span>
                <span className={styles.configValue}>{routerConfig.classifier.pii_model.model_id}</span>
              </div>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Threshold</span>
                <span className={styles.configValue}>{formatThreshold(routerConfig.classifier.pii_model.threshold)}</span>
              </div>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>ModernBERT</span>
                <span className={`${styles.statusBadge} ${routerConfig.classifier.pii_model.use_modernbert ? styles.statusActive : styles.statusInactive}`}>
                  {routerConfig.classifier.pii_model.use_modernbert ? '✓ Enabled' : '✗ Disabled'}
                </span>
              </div>
              {routerConfig.classifier.pii_model.pii_mapping_path && (
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Mapping Path</span>
                  <span className={styles.configValue}>{routerConfig.classifier.pii_model.pii_mapping_path}</span>
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
        {routerConfig.prompt_guard && !isReadonly && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
              openEditModal(
                'Edit Jailbreak Detection Configuration',
                routerConfig.prompt_guard || {},
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
        {routerConfig.prompt_guard ? (
          <div className={styles.modelCard}>
            <div className={styles.modelCardHeader}>
              <span className={styles.modelCardTitle}>Jailbreak Protection</span>
              <span className={`${styles.statusBadge} ${routerConfig.prompt_guard.enabled ? styles.statusActive : styles.statusInactive}`}>
                {routerConfig.prompt_guard.enabled ? '✓ Enabled' : '✗ Disabled'}
              </span>
            </div>
            {routerConfig.prompt_guard.enabled && (
              <div className={styles.modelCardBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Model ID</span>
                  <span className={styles.configValue}>{routerConfig.prompt_guard.model_id}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Threshold</span>
                  <span className={styles.configValue}>{formatThreshold(routerConfig.prompt_guard.threshold)}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Use CPU</span>
                  <span className={`${styles.statusBadge} ${styles.statusActive}`}>
                    {routerConfig.prompt_guard.use_cpu ? 'CPU' : 'GPU'}
                  </span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>ModernBERT</span>
                  <span className={`${styles.statusBadge} ${routerConfig.prompt_guard.use_modernbert ? styles.statusActive : styles.statusInactive}`}>
                    {routerConfig.prompt_guard.use_modernbert ? '✓ Enabled' : '✗ Disabled'}
                  </span>
                </div>
                {routerConfig.prompt_guard.jailbreak_mapping_path && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Mapping Path</span>
                    <span className={styles.configValue}>{routerConfig.prompt_guard.jailbreak_mapping_path}</span>
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
        <h3 className={styles.sectionTitle}>Similarity BERT Configuration</h3>
        {routerConfig.bert_model && !isReadonly && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
              openEditModal(
                'Edit Similarity BERT Configuration',
                routerConfig.bert_model || {},
                [
                  { name: 'model_id', label: 'Model ID', type: 'text', required: true, placeholder: 'e.g., sentence-transformers/all-MiniLM-L6-v2', description: 'HuggingFace model ID for semantic similarity' },
                  { name: 'threshold', label: 'Similarity Threshold', type: 'percentage', required: true, placeholder: '80', description: 'Minimum similarity score for cache hits (0-100%)', step: 1 },
                  { name: 'use_cpu', label: 'Use CPU', type: 'boolean', description: 'Use CPU instead of GPU for inference' },
                ],
                async (data) => {
                  const newConfig = cloneConfig(config)
                  newConfig.bert_model = data
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
        {routerConfig.bert_model ? (
          <div className={styles.modelCard}>
            <div className={styles.modelCardHeader}>
              <span className={styles.modelCardTitle}>BERT Model (Semantic Similarity)</span>
              <span className={`${styles.statusBadge} ${styles.statusActive}`}>
                {routerConfig.bert_model.use_cpu ? 'CPU' : 'GPU'}
              </span>
            </div>
            <div className={styles.modelCardBody}>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Model ID</span>
                <span className={styles.configValue}>{routerConfig.bert_model.model_id}</span>
              </div>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Threshold</span>
                <span className={styles.configValue}>{formatThreshold(routerConfig.bert_model.threshold)}</span>
              </div>
            </div>
          </div>
        ) : (
          <div className={styles.emptyState}>BERT model not configured</div>
        )}

        {routerConfig.semantic_cache && (
          <div className={styles.featureCard}>
            <div className={styles.featureHeader}>
              <span className={styles.featureTitle}>Semantic Cache</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <span className={`${styles.statusBadge} ${routerConfig.semantic_cache.enabled ? styles.statusActive : styles.statusInactive}`}>
                  {routerConfig.semantic_cache.enabled ? '✓ Enabled' : '✗ Disabled'}
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
            {routerConfig.semantic_cache.enabled && (
              <div className={styles.featureBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Backend Type</span>
                  <span className={styles.configValue}>{routerConfig.semantic_cache.backend_type || 'memory'}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Similarity Threshold</span>
                  <span className={styles.configValue}>{formatThreshold(routerConfig.semantic_cache.similarity_threshold ?? 0)}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Max Entries</span>
                  <span className={styles.configValue}>{routerConfig.semantic_cache.max_entries}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>TTL</span>
                  <span className={styles.configValue}>{routerConfig.semantic_cache.ttl_seconds}s</span>
                </div>
                {routerConfig.semantic_cache.eviction_policy && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Eviction Policy</span>
                    <span className={styles.configValue}>{routerConfig.semantic_cache.eviction_policy}</span>
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
