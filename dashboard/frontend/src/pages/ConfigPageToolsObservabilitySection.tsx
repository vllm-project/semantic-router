import styles from './ConfigPage.module.css'
import { formatThreshold } from './configPageSupport'
import { cloneConfig, type RouterToolsSectionProps } from './configPageRouterSectionSupport'

export default function ConfigPageToolsObservabilitySection({
  config,
  routerConfig,
  toolsData,
  toolsLoading,
  toolsError,
  isReadonly,
  openEditModal,
  saveConfig,
}: RouterToolsSectionProps) {
  const renderToolsConfiguration = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <h3 className={styles.sectionTitle}>Tools Configuration</h3>
        {routerConfig.tools && !isReadonly && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
              openEditModal(
                'Edit Tools Configuration',
                routerConfig.tools || {},
                [
                  { name: 'enabled', label: 'Enable Tool Auto-Selection', type: 'boolean', description: 'Enable automatic tool selection based on similarity' },
                  { name: 'top_k', label: 'Top K', type: 'number', placeholder: '3', description: 'Number of top similar tools to select' },
                  { name: 'similarity_threshold', label: 'Similarity Threshold', type: 'percentage', placeholder: '70', description: 'Minimum similarity score for tool selection (0-100%)', step: 1 },
                  { name: 'fallback_to_empty', label: 'Fallback to Empty', type: 'boolean', description: 'Return empty list if no tools meet threshold' },
                  { name: 'tools_db_path', label: 'Tools Database Path', type: 'text', placeholder: 'config/tools_db.json', description: 'Path to tools database JSON file' },
                ],
                async (data) => {
                  const newConfig = cloneConfig(config)
                  newConfig.tools = data
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
        {routerConfig.tools ? (
          <div className={styles.featureCard}>
            <div className={styles.featureHeader}>
              <span className={styles.featureTitle}>Tool Auto-Selection</span>
              <span className={`${styles.statusBadge} ${routerConfig.tools.enabled ? styles.statusActive : styles.statusInactive}`}>
                {routerConfig.tools.enabled ? '✓ Enabled' : '✗ Disabled'}
              </span>
            </div>
            {routerConfig.tools.enabled && (
              <div className={styles.featureBody}>
                <div className={styles.configRow}><span className={styles.configLabel}>Top K</span><span className={styles.configValue}>{routerConfig.tools.top_k}</span></div>
                <div className={styles.configRow}><span className={styles.configLabel}>Similarity Threshold</span><span className={styles.configValue}>{formatThreshold(routerConfig.tools.similarity_threshold)}</span></div>
                <div className={styles.configRow}><span className={styles.configLabel}>Fallback to Empty</span><span className={styles.configValue}>{routerConfig.tools.fallback_to_empty ? 'Yes' : 'No'}</span></div>
              </div>
            )}
          </div>
        ) : (
          <div className={styles.emptyState}>Tools configuration not available</div>
        )}
      </div>
    </div>
  )

  const renderToolsDB = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <h3 className={styles.sectionTitle}>Tools Database</h3>
        {toolsData.length > 0 && <span className={styles.badge}>{toolsData.length} tools</span>}
      </div>
      <div className={styles.sectionContent}>
        {config?.tools?.tools_db_path ? (
          <>
            <div className={styles.featureCard}>
              <div className={styles.featureHeader}>
                <span className={styles.featureTitle}>Database Path</span>
              </div>
              <div className={styles.featureBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Path</span>
                  <span className={styles.configValue}>{config.tools.tools_db_path}</span>
                </div>
              </div>
            </div>
            {toolsLoading && <div className={styles.loadingState}>Loading tools...</div>}
            {toolsError && <div className={styles.errorState}>Error loading tools: {toolsError}</div>}
            {!toolsLoading && !toolsError && toolsData.length > 0 && (
              <div className={styles.toolsGrid}>
                {toolsData.map((tool, index) => (
                  <div key={index} className={styles.toolCard}>
                    <div className={styles.toolHeader}>
                      <span className={styles.toolName}>{tool.tool.function.name}</span>
                      {tool.category && <span className={`${styles.badge} ${styles.badgeInfo}`}>{tool.category}</span>}
                    </div>
                    <div className={styles.toolFunctionDescription}><strong>Function:</strong> {tool.tool.function.description}</div>
                    {tool.description && tool.description !== tool.tool.function.description && (
                      <div className={styles.toolSimilarityDescription}>
                        <div className={styles.similarityDescriptionLabel}>Similarity Keywords</div>
                        <div className={styles.similarityDescriptionText}>{tool.description}</div>
                      </div>
                    )}
                    {tool.tool.function.parameters.properties && (
                      <div className={styles.toolParameters}>
                        <div className={styles.toolParametersHeader}>Parameters:</div>
                        {Object.entries(tool.tool.function.parameters.properties).map(([paramName, paramInfo]: [string, { type?: string; description?: string }]) => (
                          <div key={paramName} className={styles.toolParameter}>
                            <div>
                              <span className={styles.parameterName}>
                                {paramName}
                                {tool.tool.function.parameters.required?.includes(paramName) && <span className={styles.requiredBadge}>*</span>}
                              </span>
                              <span className={styles.parameterType}>{paramInfo.type}</span>
                            </div>
                            {paramInfo.description && <div className={styles.parameterDescription}>{paramInfo.description}</div>}
                          </div>
                        ))}
                      </div>
                    )}
                    {tool.tags && tool.tags.length > 0 && (
                      <div className={styles.toolTags}>
                        {tool.tags.map((tag, idx) => <span key={idx} className={styles.toolTag}>{tag}</span>)}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </>
        ) : (
          <div className={styles.emptyState}>Tools database path not configured</div>
        )}
      </div>
    </div>
  )

  const renderObservabilityTracing = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <h3 className={styles.sectionTitle}>Distributed Tracing</h3>
        {routerConfig.observability?.tracing && !isReadonly && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
              openEditModal(
                'Edit Distributed Tracing Configuration',
                routerConfig.observability?.tracing || {},
                [
                  { name: 'enabled', label: 'Enable Tracing', type: 'boolean', description: 'Enable distributed tracing' },
                  { name: 'provider', label: 'Provider', type: 'select', options: ['jaeger', 'zipkin', 'otlp'], description: 'Tracing provider' },
                  { name: 'exporter', label: 'Exporter Configuration (JSON)', type: 'json', placeholder: '{\"type\": \"otlp\", \"endpoint\": \"http://localhost:4318\"}', description: 'Exporter configuration as JSON object' },
                  { name: 'sampling', label: 'Sampling Configuration (JSON)', type: 'json', placeholder: '{\"type\": \"probabilistic\", \"rate\": 0.1}', description: 'Sampling configuration as JSON object' },
                  { name: 'resource', label: 'Resource Configuration (JSON)', type: 'json', placeholder: '{\"service_name\": \"semantic-router\", \"service_version\": \"1.0.0\", \"deployment_environment\": \"production\"}', description: 'Resource attributes as JSON object' },
                ],
                async (data) => {
                  const newConfig = cloneConfig(config)
                  if (!newConfig.observability) newConfig.observability = {}
                  newConfig.observability.tracing = data
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
        {routerConfig.observability?.tracing ? (
          <div className={styles.featureCard}>
            <div className={styles.featureHeader}>
              <span className={styles.featureTitle}>Tracing Status</span>
              <span className={`${styles.statusBadge} ${routerConfig.observability.tracing.enabled ? styles.statusActive : styles.statusInactive}`}>
                {routerConfig.observability.tracing.enabled ? '✓ Enabled' : '✗ Disabled'}
              </span>
            </div>
            {routerConfig.observability.tracing.enabled && (
              <div className={styles.featureBody}>
                <div className={styles.configRow}><span className={styles.configLabel}>Provider</span><span className={styles.configValue}>{routerConfig.observability.tracing.provider}</span></div>
                <div className={styles.configRow}><span className={styles.configLabel}>Exporter Type</span><span className={styles.configValue}>{routerConfig.observability.tracing.exporter?.type}</span></div>
                {routerConfig.observability.tracing.exporter?.endpoint && <div className={styles.configRow}><span className={styles.configLabel}>Endpoint</span><span className={styles.configValue}>{routerConfig.observability.tracing.exporter.endpoint}</span></div>}
                <div className={styles.configRow}><span className={styles.configLabel}>Sampling Type</span><span className={styles.configValue}>{routerConfig.observability.tracing.sampling?.type}</span></div>
                {routerConfig.observability.tracing.sampling?.rate !== undefined && <div className={styles.configRow}><span className={styles.configLabel}>Sampling Rate</span><span className={styles.configValue}>{((routerConfig.observability.tracing.sampling.rate ?? 0) * 100).toFixed(0)}%</span></div>}
                <div className={styles.configRow}><span className={styles.configLabel}>Service Name</span><span className={styles.configValue}>{routerConfig.observability.tracing.resource?.service_name}</span></div>
                <div className={styles.configRow}><span className={styles.configLabel}>Service Version</span><span className={styles.configValue}>{routerConfig.observability.tracing.resource?.service_version}</span></div>
                <div className={styles.configRow}><span className={styles.configLabel}>Environment</span><span className={`${styles.badge} ${styles[`badge${routerConfig.observability.tracing.resource?.deployment_environment ?? ''}`]}`}>{routerConfig.observability.tracing.resource?.deployment_environment}</span></div>
              </div>
            )}
          </div>
        ) : (
          <div className={styles.emptyState}>Tracing not configured</div>
        )}
      </div>
    </div>
  )

  const renderClassificationAPI = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <h3 className={styles.sectionTitle}>Batch Classification API</h3>
        {routerConfig.api?.batch_classification && !isReadonly && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
              openEditModal(
                'Edit Batch Classification API Configuration',
                routerConfig.api?.batch_classification || {},
                [
                  { name: 'max_batch_size', label: 'Max Batch Size', type: 'number', required: true, placeholder: '100', description: 'Maximum number of items in a single batch' },
                  { name: 'concurrency_threshold', label: 'Concurrency Threshold', type: 'number', placeholder: '10', description: 'Threshold to trigger concurrent processing' },
                  { name: 'max_concurrency', label: 'Max Concurrency', type: 'number', placeholder: '5', description: 'Maximum number of concurrent batch processes' },
                  { name: 'metrics', label: 'Metrics Configuration (JSON)', type: 'json', placeholder: '{\"enabled\": true, \"sample_rate\": 0.1, \"detailed_goroutine_tracking\": false, \"high_resolution_timing\": true}', description: 'Metrics collection configuration as JSON object' },
                ],
                async (data) => {
                  const newConfig = cloneConfig(config)
                  if (!newConfig.api) newConfig.api = {}
                  newConfig.api.batch_classification = data
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
        {routerConfig.api?.batch_classification ? (
          <>
            <div className={styles.featureCard}>
              <div className={styles.featureHeader}><span className={styles.featureTitle}>Batch Configuration</span></div>
              <div className={styles.featureBody}>
                <div className={styles.configRow}><span className={styles.configLabel}>Max Batch Size</span><span className={styles.configValue}>{routerConfig.api.batch_classification.max_batch_size}</span></div>
                {routerConfig.api.batch_classification.concurrency_threshold !== undefined && <div className={styles.configRow}><span className={styles.configLabel}>Concurrency Threshold</span><span className={styles.configValue}>{routerConfig.api.batch_classification.concurrency_threshold}</span></div>}
                {routerConfig.api.batch_classification.max_concurrency !== undefined && <div className={styles.configRow}><span className={styles.configLabel}>Max Concurrency</span><span className={styles.configValue}>{routerConfig.api.batch_classification.max_concurrency}</span></div>}
              </div>
            </div>

            {routerConfig.api.batch_classification.metrics && (
              <div className={styles.featureCard}>
                <div className={styles.featureHeader}>
                  <span className={styles.featureTitle}>Metrics Collection</span>
                  <span className={`${styles.statusBadge} ${routerConfig.api.batch_classification.metrics.enabled ? styles.statusActive : styles.statusInactive}`}>
                    {routerConfig.api.batch_classification.metrics.enabled ? '✓ Enabled' : '✗ Disabled'}
                  </span>
                </div>
                {routerConfig.api.batch_classification.metrics.enabled && (
                  <div className={styles.featureBody}>
                    {routerConfig.api.batch_classification.metrics.sample_rate !== undefined && <div className={styles.configRow}><span className={styles.configLabel}>Sample Rate</span><span className={styles.configValue}>{((routerConfig.api.batch_classification.metrics.sample_rate ?? 0) * 100).toFixed(0)}%</span></div>}
                    {routerConfig.api.batch_classification.metrics.detailed_goroutine_tracking !== undefined && <div className={styles.configRow}><span className={styles.configLabel}>Goroutine Tracking</span><span className={styles.configValue}>{routerConfig.api.batch_classification.metrics.detailed_goroutine_tracking ? 'Yes' : 'No'}</span></div>}
                    {routerConfig.api.batch_classification.metrics.high_resolution_timing !== undefined && <div className={styles.configRow}><span className={styles.configLabel}>High Resolution Timing</span><span className={styles.configValue}>{routerConfig.api.batch_classification.metrics.high_resolution_timing ? 'Yes' : 'No'}</span></div>}
                  </div>
                )}
              </div>
            )}
          </>
        ) : (
          <div className={styles.emptyState}>Batch classification API not configured</div>
        )}
      </div>
    </div>
  )

  return (
    <>
      {renderToolsConfiguration()}
      {renderToolsDB()}
      {renderObservabilityTracing()}
      {renderClassificationAPI()}
    </>
  )
}
