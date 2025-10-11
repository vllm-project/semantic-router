import React, { useState, useEffect } from 'react'
import styles from './ConfigPage.module.css'
import ConfigNav, { ConfigSection } from '../components/ConfigNav'

interface VLLMEndpoint {
  name: string
  address: string
  port: number
  models: string[]
  weight: number
  health_check_path: string
}

interface ModelConfig {
  model_id: string
  use_modernbert?: boolean
  threshold: number
  use_cpu: boolean
  category_mapping_path?: string
  pii_mapping_path?: string
  jailbreak_mapping_path?: string
}

interface ModelScore {
  model: string
  score: number
  use_reasoning: boolean
}

interface Category {
  name: string
  use_reasoning: boolean
  reasoning_description: string
  reasoning_effort: string
  model_scores: ModelScore[]
}

interface ReasoningFamily {
  type: string
  parameter: string
}

interface ModelConfigEntry {
  reasoning_family?: string
  preferred_endpoints?: string[]
  pii_policy?: {
    allow_by_default: boolean
  }
}

interface TracingConfig {
  enabled: boolean
  provider: string
  exporter: {
    type: string
    endpoint?: string
    insecure?: boolean
  }
  sampling: {
    type: string
    rate?: number
  }
  resource: {
    service_name: string
    service_version: string
    deployment_environment: string
  }
}

interface APIConfig {
  batch_classification?: {
    max_batch_size: number
    concurrency_threshold: number
    max_concurrency: number
    metrics?: {
      enabled: boolean
      detailed_goroutine_tracking?: boolean
      high_resolution_timing?: boolean
      sample_rate?: number
      duration_buckets?: number[]
      size_buckets?: number[]
    }
  }
}

interface ConfigData {
  bert_model?: ModelConfig
  semantic_cache?: {
    enabled: boolean
    backend_type?: string
    similarity_threshold: number
    max_entries: number
    ttl_seconds: number
    eviction_policy?: string
  }
  tools?: {
    enabled: boolean
    top_k: number
    similarity_threshold: number
    tools_db_path: string
    fallback_to_empty: boolean
  }
  prompt_guard?: ModelConfig & { enabled: boolean }
  vllm_endpoints?: VLLMEndpoint[]
  classifier?: {
    category_model?: ModelConfig
    pii_model?: ModelConfig
  }
  categories?: Category[]
  default_reasoning_effort?: string
  default_model?: string
  model_config?: Record<string, ModelConfigEntry>
  reasoning_families?: Record<string, ReasoningFamily>
  api?: APIConfig
  observability?: {
    tracing?: TracingConfig
  }
  [key: string]: unknown
}

const ConfigPage: React.FC = () => {
  const [config, setConfig] = useState<ConfigData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedView, setSelectedView] = useState<'structured' | 'raw'>('structured')
  const [activeSection, setActiveSection] = useState<ConfigSection>('models')

  useEffect(() => {
    fetchConfig()
  }, [])

  const fetchConfig = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/router/config/all')
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      const data = await response.json()
      setConfig(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch config')
      setConfig(null)
    } finally {
      setLoading(false)
    }
  }

  const handleRefresh = () => {
    fetchConfig()
  }

  const renderBackendEndpoints = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionIcon}>🔌</span>
        <h3 className={styles.sectionTitle}>Backend Endpoints</h3>
        <span className={styles.badge}>{config?.vllm_endpoints?.length || 0} endpoints</span>
      </div>
      <div className={styles.sectionContent}>
        {config?.vllm_endpoints?.map((endpoint, index) => (
          <div key={index} className={styles.endpointCard}>
            <div className={styles.endpointHeader}>
              <span className={styles.endpointName}>{endpoint.name}</span>
              <span className={styles.badge}>{endpoint.address}:{endpoint.port}</span>
            </div>
            <div className={styles.endpointDetails}>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Models</span>
                <div className={styles.modelTags}>
                  {endpoint.models.map((model, idx) => (
                    <span key={idx} className={styles.modelTag}>{model}</span>
                  ))}
                </div>
              </div>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Weight</span>
                <span className={styles.configValue}>{endpoint.weight}</span>
              </div>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Health Check</span>
                <span className={styles.configValue}>{endpoint.health_check_path}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )

  const renderAIModels = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionIcon}>🤖</span>
        <h3 className={styles.sectionTitle}>AI Models Configuration</h3>
      </div>
      <div className={styles.sectionContent}>
        {config?.bert_model && (
          <div className={styles.modelCard}>
            <div className={styles.modelCardHeader}>
              <span className={styles.modelCardTitle}>BERT Model (Semantic Similarity)</span>
              <span className={`${styles.statusBadge} ${styles.statusActive}`}>
                {config.bert_model.use_cpu ? '💻 CPU' : '🎮 GPU'}
              </span>
            </div>
            <div className={styles.modelCardBody}>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Model ID</span>
                <span className={styles.configValue}>{config.bert_model.model_id}</span>
              </div>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Threshold</span>
                <span className={styles.configValue}>{config.bert_model.threshold}</span>
              </div>
            </div>
          </div>
        )}

        {config?.classifier?.category_model && (
          <div className={styles.modelCard}>
            <div className={styles.modelCardHeader}>
              <span className={styles.modelCardTitle}>Category Classifier</span>
              <span className={`${styles.statusBadge} ${styles.statusActive}`}>
                {config.classifier.category_model.use_cpu ? '💻 CPU' : '🎮 GPU'}
              </span>
            </div>
            <div className={styles.modelCardBody}>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Model ID</span>
                <span className={styles.configValue}>{config.classifier.category_model.model_id}</span>
              </div>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Threshold</span>
                <span className={styles.configValue}>{config.classifier.category_model.threshold}</span>
              </div>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>ModernBERT</span>
                <span className={`${styles.statusBadge} ${config.classifier.category_model.use_modernbert ? styles.statusActive : styles.statusInactive}`}>
                  {config.classifier.category_model.use_modernbert ? '✓ Enabled' : '✗ Disabled'}
                </span>
              </div>
            </div>
          </div>
        )}

        {config?.classifier?.pii_model && (
          <div className={styles.modelCard}>
            <div className={styles.modelCardHeader}>
              <span className={styles.modelCardTitle}>PII Detector</span>
              <span className={`${styles.statusBadge} ${styles.statusActive}`}>
                {config.classifier.pii_model.use_cpu ? '💻 CPU' : '🎮 GPU'}
              </span>
            </div>
            <div className={styles.modelCardBody}>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Model ID</span>
                <span className={styles.configValue}>{config.classifier.pii_model.model_id}</span>
              </div>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Threshold</span>
                <span className={styles.configValue}>{config.classifier.pii_model.threshold}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )

  const renderCategories = () => (
    <div className={`${styles.section} ${styles.categoriesSection}`}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionIcon}>📊</span>
        <h3 className={styles.sectionTitle}>Categories & Routing</h3>
        <span className={styles.badge}>{config?.categories?.length || 0} categories</span>
      </div>
      <div className={styles.sectionContent}>
        {/* Core Settings at the top */}
        <div className={styles.coreSettingsInline}>
          <div className={styles.inlineConfigRow}>
            <span className={styles.inlineConfigLabel}>🎯 Default Model:</span>
            <span className={styles.inlineConfigValue}>{config?.default_model || 'N/A'}</span>
          </div>
          <div className={styles.inlineConfigRow}>
            <span className={styles.inlineConfigLabel}>⚡ Default Reasoning Effort:</span>
            <span className={`${styles.badge} ${styles[`badge${config?.default_reasoning_effort || 'medium'}`]}`}>
              {config?.default_reasoning_effort || 'medium'}
            </span>
          </div>
        </div>

        <div className={styles.categoryGrid}>
          {config?.categories?.map((category, index) => (
            <div key={index} className={styles.categoryCard}>
              <div className={styles.categoryHeader}>
                <span className={styles.categoryName}>{category.name}</span>
                {category.use_reasoning && (
                  <span className={`${styles.reasoningBadge} ${styles[`reasoning${category.reasoning_effort}`]}`}>
                    ⚡ {category.reasoning_effort}
                  </span>
                )}
              </div>
              <p className={styles.categoryDescription}>{category.reasoning_description}</p>
              <div className={styles.categoryModels}>
                <div className={styles.categoryModelsHeader}>Top Models</div>
                {category.model_scores.slice(0, 3).map((modelScore, idx) => (
                  <div key={idx} className={styles.modelScoreRow}>
                    <span className={styles.modelScoreName}>
                      {modelScore.model}
                      {modelScore.use_reasoning && <span className={styles.reasoningIcon}>🧠</span>}
                    </span>
                    <div className={styles.scoreBar}>
                      <div
                        className={styles.scoreBarFill}
                        style={{ width: `${modelScore.score * 100}%` }}
                      ></div>
                      <span className={styles.scoreText}>{(modelScore.score * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )

  const renderSecurity = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionIcon}>🛡️</span>
        <h3 className={styles.sectionTitle}>Security Features</h3>
      </div>
      <div className={styles.sectionContent}>
        {config?.prompt_guard && (
          <div className={styles.featureCard}>
            <div className={styles.featureHeader}>
              <span className={styles.featureTitle}>Jailbreak Protection</span>
              <span className={`${styles.statusBadge} ${config.prompt_guard.enabled ? styles.statusActive : styles.statusInactive}`}>
                {config.prompt_guard.enabled ? '✓ Enabled' : '✗ Disabled'}
              </span>
            </div>
            {config.prompt_guard.enabled && (
              <div className={styles.featureBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Model</span>
                  <span className={styles.configValue}>{config.prompt_guard.model_id}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Threshold</span>
                  <span className={styles.configValue}>{config.prompt_guard.threshold}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>ModernBERT</span>
                  <span className={styles.configValue}>{config.prompt_guard.use_modernbert ? 'Yes' : 'No'}</span>
                </div>
              </div>
            )}
          </div>
        )}

        {config?.classifier?.pii_model && (
          <div className={styles.featureCard}>
            <div className={styles.featureHeader}>
              <span className={styles.featureTitle}>PII Detection</span>
              <span className={`${styles.statusBadge} ${styles.statusActive}`}>✓ Configured</span>
            </div>
            <div className={styles.featureBody}>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Threshold</span>
                <span className={styles.configValue}>{config.classifier.pii_model.threshold}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )

  // Section-specific renders
  const renderModelsSection = () => (
    <div className={styles.sectionPanel}>
      {renderBackendEndpoints()}
      {renderAIModels()}
    </div>
  )

  const renderPromptGuardSection = () => (
    <div className={styles.sectionPanel}>
      {renderSecurity()}
    </div>
  )

  const renderSimilarityCacheSection = () => (
    <div className={styles.sectionPanel}>
      {config?.semantic_cache && (
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionIcon}>⚡</span>
            <h3 className={styles.sectionTitle}>Semantic Cache Configuration</h3>
          </div>
          <div className={styles.sectionContent}>
            <div className={styles.featureCard}>
              <div className={styles.featureHeader}>
                <span className={styles.featureTitle}>Status</span>
                <span className={`${styles.statusBadge} ${config.semantic_cache.enabled ? styles.statusActive : styles.statusInactive}`}>
                  {config.semantic_cache.enabled ? '✓ Enabled' : '✗ Disabled'}
                </span>
              </div>
              {config.semantic_cache.enabled && (
                <div className={styles.featureBody}>
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Similarity Threshold</span>
                    <span className={styles.configValue}>{config.semantic_cache.similarity_threshold}</span>
                  </div>
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Max Entries</span>
                    <span className={styles.configValue}>{config.semantic_cache.max_entries}</span>
                  </div>
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>TTL</span>
                    <span className={styles.configValue}>{config.semantic_cache.ttl_seconds}s</span>
                  </div>
                </div>
              )}
            </div>

            {config?.bert_model && (
              <div className={styles.modelCard}>
                <div className={styles.modelCardHeader}>
                  <span className={styles.modelCardTitle}>BERT Model (Similarity)</span>
                  <span className={`${styles.statusBadge} ${styles.statusActive}`}>
                    {config.bert_model.use_cpu ? '💻 CPU' : '🎮 GPU'}
                  </span>
                </div>
                <div className={styles.modelCardBody}>
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Model ID</span>
                    <span className={styles.configValue}>{config.bert_model.model_id}</span>
                  </div>
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Threshold</span>
                    <span className={styles.configValue}>{config.bert_model.threshold}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )

  const renderIntelligentRoutingSection = () => (
    <div className={styles.sectionPanel}>
      {/* Model Configuration */}
      {config?.model_config && Object.keys(config.model_config).length > 0 && (
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionIcon}>⚙️</span>
            <h3 className={styles.sectionTitle}>Model Configuration</h3>
            <span className={styles.badge}>{Object.keys(config.model_config).length} models</span>
          </div>
          <div className={styles.sectionContent}>
            <div className={styles.modelConfigGrid}>
              {Object.entries(config.model_config).map(([modelName, modelConfig]) => (
                <div key={modelName} className={styles.modelConfigCard}>
                  <div className={styles.modelConfigHeader}>
                    <span className={styles.modelConfigName}>{modelName}</span>
                  </div>
                  <div className={styles.modelConfigBody}>
                    {modelConfig.reasoning_family && (
                      <div className={styles.configRow}>
                        <span className={styles.configLabel}>🧠 Reasoning Family</span>
                        <span className={`${styles.badge} ${styles.badgeInfo}`}>
                          {modelConfig.reasoning_family}
                        </span>
                      </div>
                    )}
                    {modelConfig.preferred_endpoints && modelConfig.preferred_endpoints.length > 0 && (
                      <div className={styles.configRow}>
                        <span className={styles.configLabel}>🔌 Preferred Endpoints</span>
                        <div className={styles.endpointTags}>
                          {modelConfig.preferred_endpoints.map((endpoint, idx) => (
                            <span key={idx} className={styles.endpointTag}>{endpoint}</span>
                          ))}
                        </div>
                      </div>
                    )}
                    {modelConfig.pii_policy && (
                      <div className={styles.configRow}>
                        <span className={styles.configLabel}>🔒 PII Policy</span>
                        <span className={`${styles.statusBadge} ${modelConfig.pii_policy.allow_by_default ? styles.statusActive : styles.statusInactive}`}>
                          {modelConfig.pii_policy.allow_by_default ? 'Allow by default' : 'Block by default'}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Reasoning Families */}
      {config?.reasoning_families && Object.keys(config.reasoning_families).length > 0 && (
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionIcon}>🧠</span>
            <h3 className={styles.sectionTitle}>Reasoning Families</h3>
            <span className={styles.badge}>{Object.keys(config.reasoning_families).length} families</span>
          </div>
          <div className={styles.sectionContent}>
            <div className={styles.reasoningFamiliesGrid}>
              {Object.entries(config.reasoning_families).map(([familyName, familyConfig]) => (
                <div key={familyName} className={styles.reasoningFamilyCard}>
                  <div className={styles.reasoningFamilyHeader}>
                    <span className={styles.reasoningFamilyName}>{familyName}</span>
                  </div>
                  <div className={styles.reasoningFamilyBody}>
                    <div className={styles.configRow}>
                      <span className={styles.configLabel}>Type</span>
                      <span className={styles.configValue}>{familyConfig.type}</span>
                    </div>
                    <div className={styles.configRow}>
                      <span className={styles.configLabel}>Parameter</span>
                      <span className={styles.configValue}><code>{familyConfig.parameter}</code></span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Categories */}
      {renderCategories()}
    </div>
  )

  const renderToolsSelectionSection = () => (
    <div className={styles.sectionPanel}>
      {config?.tools && (
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionIcon}>🔧</span>
            <h3 className={styles.sectionTitle}>Tool Auto-Selection</h3>
          </div>
          <div className={styles.sectionContent}>
            <div className={styles.featureCard}>
              <div className={styles.featureHeader}>
                <span className={styles.featureTitle}>Status</span>
                <span className={`${styles.statusBadge} ${config.tools.enabled ? styles.statusActive : styles.statusInactive}`}>
                  {config.tools.enabled ? '✓ Enabled' : '✗ Disabled'}
                </span>
              </div>
              {config.tools.enabled && (
                <div className={styles.featureBody}>
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Top K</span>
                    <span className={styles.configValue}>{config.tools.top_k}</span>
                  </div>
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Similarity Threshold</span>
                    <span className={styles.configValue}>{config.tools.similarity_threshold}</span>
                  </div>
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Tools Database Path</span>
                    <span className={styles.configValue}>{config.tools.tools_db_path}</span>
                  </div>
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Fallback to Empty</span>
                    <span className={styles.configValue}>{config.tools.fallback_to_empty ? 'Yes' : 'No'}</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )

  const renderObservabilitySection = () => (
    <div className={styles.sectionPanel}>
      {/* Distributed Tracing */}
      {config?.observability?.tracing && (
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionIcon}>�</span>
            <h3 className={styles.sectionTitle}>Distributed Tracing</h3>
          </div>
          <div className={styles.sectionContent}>
            <div className={styles.featureCard}>
              <div className={styles.featureHeader}>
                <span className={styles.featureTitle}>Tracing Status</span>
                <span className={`${styles.statusBadge} ${config.observability.tracing.enabled ? styles.statusActive : styles.statusInactive}`}>
                  {config.observability.tracing.enabled ? '✓ Enabled' : '✗ Disabled'}
                </span>
              </div>
              {config.observability.tracing.enabled && (
                <div className={styles.featureBody}>
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Provider</span>
                    <span className={styles.configValue}>{config.observability.tracing.provider}</span>
                  </div>
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Exporter Type</span>
                    <span className={styles.configValue}>{config.observability.tracing.exporter.type}</span>
                  </div>
                  {config.observability.tracing.exporter.endpoint && (
                    <div className={styles.configRow}>
                      <span className={styles.configLabel}>Endpoint</span>
                      <span className={styles.configValue}>{config.observability.tracing.exporter.endpoint}</span>
                    </div>
                  )}
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Sampling Type</span>
                    <span className={styles.configValue}>{config.observability.tracing.sampling.type}</span>
                  </div>
                  {config.observability.tracing.sampling.rate !== undefined && (
                    <div className={styles.configRow}>
                      <span className={styles.configLabel}>Sampling Rate</span>
                      <span className={styles.configValue}>{(config.observability.tracing.sampling.rate * 100).toFixed(0)}%</span>
                    </div>
                  )}
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Service Name</span>
                    <span className={styles.configValue}>{config.observability.tracing.resource.service_name}</span>
                  </div>
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Service Version</span>
                    <span className={styles.configValue}>{config.observability.tracing.resource.service_version}</span>
                  </div>
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Environment</span>
                    <span className={`${styles.badge} ${styles[`badge${config.observability.tracing.resource.deployment_environment}`]}`}>
                      {config.observability.tracing.resource.deployment_environment}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* API & Metrics */}
      {config?.api?.batch_classification && (
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionIcon}>📊</span>
            <h3 className={styles.sectionTitle}>API & Performance Metrics</h3>
          </div>
          <div className={styles.sectionContent}>
            <div className={styles.featureCard}>
              <div className={styles.featureHeader}>
                <span className={styles.featureTitle}>Batch Classification</span>
              </div>
              <div className={styles.featureBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Max Batch Size</span>
                  <span className={styles.configValue}>{config.api.batch_classification.max_batch_size}</span>
                </div>
                {config.api.batch_classification.concurrency_threshold !== undefined && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Concurrency Threshold</span>
                    <span className={styles.configValue}>{config.api.batch_classification.concurrency_threshold}</span>
                  </div>
                )}
                {config.api.batch_classification.max_concurrency !== undefined && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Max Concurrency</span>
                    <span className={styles.configValue}>{config.api.batch_classification.max_concurrency}</span>
                  </div>
                )}
              </div>
            </div>

            {config.api.batch_classification.metrics && (
              <div className={styles.featureCard}>
                <div className={styles.featureHeader}>
                  <span className={styles.featureTitle}>Metrics Collection</span>
                  <span className={`${styles.statusBadge} ${config.api.batch_classification.metrics.enabled ? styles.statusActive : styles.statusInactive}`}>
                    {config.api.batch_classification.metrics.enabled ? '✓ Enabled' : '✗ Disabled'}
                  </span>
                </div>
                {config.api.batch_classification.metrics.enabled && (
                  <div className={styles.featureBody}>
                    {config.api.batch_classification.metrics.sample_rate !== undefined && (
                      <div className={styles.configRow}>
                        <span className={styles.configLabel}>Sample Rate</span>
                        <span className={styles.configValue}>{(config.api.batch_classification.metrics.sample_rate * 100).toFixed(0)}%</span>
                      </div>
                    )}
                    {config.api.batch_classification.metrics.detailed_goroutine_tracking !== undefined && (
                      <div className={styles.configRow}>
                        <span className={styles.configLabel}>Goroutine Tracking</span>
                        <span className={styles.configValue}>{config.api.batch_classification.metrics.detailed_goroutine_tracking ? 'Yes' : 'No'}</span>
                      </div>
                    )}
                    {config.api.batch_classification.metrics.high_resolution_timing !== undefined && (
                      <div className={styles.configRow}>
                        <span className={styles.configLabel}>High Resolution Timing</span>
                        <span className={styles.configValue}>{config.api.batch_classification.metrics.high_resolution_timing ? 'Yes' : 'No'}</span>
                      </div>
                    )}
                    <div className={styles.configRow}>
                      <span className={styles.configLabel}>Metrics Endpoint</span>
                      <span className={styles.configValue}><code>/metrics</code></span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Link to monitoring dashboard */}
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionIcon}>📈</span>
          <h3 className={styles.sectionTitle}>Monitoring Dashboard</h3>
        </div>
        <div className={styles.sectionContent}>
          <div className={styles.observabilityInfo}>
            <p>📊 View real-time metrics and performance data in the <strong>Monitoring</strong> tab</p>
            <p>🔍 Distributed traces help diagnose latency and errors across services</p>
            <p>⚡ Performance metrics are collected for optimization insights</p>
          </div>
        </div>
      </div>
    </div>
  )

  const renderActiveSection = () => {
    switch (activeSection) {
      case 'models':
        return renderModelsSection()
      case 'prompt-guard':
        return renderPromptGuardSection()
      case 'similarity-cache':
        return renderSimilarityCacheSection()
      case 'intelligent-routing':
        return renderIntelligentRoutingSection()
      case 'tools-selection':
        return renderToolsSelectionSection()
      case 'observability':
        return renderObservabilitySection()
      default:
        return renderModelsSection()
    }
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h2 className={styles.title}>⚙️ Configuration</h2>
          <div className={styles.viewToggle}>
            <button
              className={`${styles.toggleButton} ${selectedView === 'structured' ? styles.active : ''}`}
              onClick={() => setSelectedView('structured')}
            >
              📋 Structured
            </button>
            <button
              className={`${styles.toggleButton} ${selectedView === 'raw' ? styles.active : ''}`}
              onClick={() => setSelectedView('raw')}
            >
              💻 Raw YAML
            </button>
          </div>
        </div>
        <button onClick={handleRefresh} className={styles.button} disabled={loading}>
          🔄 Refresh
        </button>
      </div>

      <div className={styles.content}>
        {loading && (
          <div className={styles.loading}>
            <div className={styles.spinner}></div>
            <p>Loading configuration...</p>
          </div>
        )}

        {error && !loading && (
          <div className={styles.error}>
            <span className={styles.errorIcon}>⚠️</span>
            <div>
              <h3>Error Loading Config</h3>
              <p>{error}</p>
            </div>
          </div>
        )}

        {config && !loading && !error && (
          <>
            {selectedView === 'structured' ? (
              <div className={styles.mainLayout}>
                <ConfigNav
                  activeSection={activeSection}
                  onSectionChange={setActiveSection}
                />
                <div className={styles.contentArea}>
                  {renderActiveSection()}
                </div>
              </div>
            ) : (
              <pre className={styles.codeBlock}>
                <code>{JSON.stringify(config, null, 2)}</code>
              </pre>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default ConfigPage
