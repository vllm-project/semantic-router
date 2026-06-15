import styles from './ConfigPage.module.css'
import type { ToolIntegrationConfig, TracingConfig } from './configPageSupport'
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
  const toolsConfig = routerConfig.tools
  const tracingConfig = routerConfig.observability?.tracing
  const batchClassification = routerConfig.api?.batch_classification
  const batchMetrics = batchClassification?.metrics

  const renderToolsConfiguration = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <h3 className={styles.sectionTitle}>Tools Configuration</h3>
        {toolsConfig && !isReadonly && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
	              openEditModal<ToolIntegrationConfig>(
	                'Edit Tools Configuration',
	                toolsConfig,
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
	        {toolsConfig ? (
          <div className={styles.featureCard}>
            <div className={styles.featureHeader}>
              <span className={styles.featureTitle}>Tool Auto-Selection</span>
	              <span className={`${styles.statusBadge} ${toolsConfig.enabled ? styles.statusActive : styles.statusInactive}`}>
	                {toolsConfig.enabled ? '✓ Enabled' : '✗ Disabled'}
	              </span>
	            </div>
	            {toolsConfig.enabled && (
	              <div className={styles.featureBody}>
	                <div className={styles.configRow}><span className={styles.configLabel}>Top K</span><span className={styles.configValue}>{toolsConfig.top_k}</span></div>
	                <div className={styles.configRow}><span className={styles.configLabel}>Similarity Threshold</span><span className={styles.configValue}>{formatThreshold(toolsConfig.similarity_threshold ?? 0)}</span></div>
	                <div className={styles.configRow}><span className={styles.configLabel}>Fallback to Empty</span><span className={styles.configValue}>{toolsConfig.fallback_to_empty ? 'Yes' : 'No'}</span></div>
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
        {tracingConfig && !isReadonly && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
	              openEditModal<TracingConfig>(
	                'Edit Distributed Tracing Configuration',
	                tracingConfig,
                [
                  { name: 'enabled', label: 'Enable Tracing', type: 'boolean', description: 'Enable distributed tracing' },
                  { name: 'provider', label: 'Provider', type: 'select', options: ['jaeger', 'zipkin', 'otlp'], description: 'Tracing provider' },
	                  { name: 'exporter', label: 'Exporter Configuration (JSON)', type: 'json', placeholder: '{"type": "otlp", "endpoint": "http://localhost:4318"}', description: 'Exporter configuration as JSON object' },
	                  { name: 'sampling', label: 'Sampling Configuration (JSON)', type: 'json', placeholder: '{"type": "probabilistic", "rate": 0.1}', description: 'Sampling configuration as JSON object' },
	                  { name: 'resource', label: 'Resource Configuration (JSON)', type: 'json', placeholder: '{"service_name": "semantic-router", "service_version": "1.0.0", "deployment_environment": "production"}', description: 'Resource attributes as JSON object' },
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
	        {tracingConfig ? (
          <div className={styles.featureCard}>
            <div className={styles.featureHeader}>
              <span className={styles.featureTitle}>Tracing Status</span>
	              <span className={`${styles.statusBadge} ${tracingConfig.enabled ? styles.statusActive : styles.statusInactive}`}>
	                {tracingConfig.enabled ? '✓ Enabled' : '✗ Disabled'}
	              </span>
	            </div>
	            {tracingConfig.enabled && (
	              <div className={styles.featureBody}>
	                <div className={styles.configRow}><span className={styles.configLabel}>Provider</span><span className={styles.configValue}>{tracingConfig.provider}</span></div>
	                <div className={styles.configRow}><span className={styles.configLabel}>Exporter Type</span><span className={styles.configValue}>{tracingConfig.exporter?.type}</span></div>
	                {tracingConfig.exporter?.endpoint && <div className={styles.configRow}><span className={styles.configLabel}>Endpoint</span><span className={styles.configValue}>{tracingConfig.exporter.endpoint}</span></div>}
	                <div className={styles.configRow}><span className={styles.configLabel}>Sampling Type</span><span className={styles.configValue}>{tracingConfig.sampling?.type}</span></div>
	                {tracingConfig.sampling?.rate !== undefined && <div className={styles.configRow}><span className={styles.configLabel}>Sampling Rate</span><span className={styles.configValue}>{((tracingConfig.sampling.rate ?? 0) * 100).toFixed(0)}%</span></div>}
	                <div className={styles.configRow}><span className={styles.configLabel}>Service Name</span><span className={styles.configValue}>{tracingConfig.resource?.service_name}</span></div>
	                <div className={styles.configRow}><span className={styles.configLabel}>Service Version</span><span className={styles.configValue}>{tracingConfig.resource?.service_version}</span></div>
	                <div className={styles.configRow}><span className={styles.configLabel}>Environment</span><span className={`${styles.badge} ${styles[`badge${tracingConfig.resource?.deployment_environment ?? ''}`]}`}>{tracingConfig.resource?.deployment_environment}</span></div>
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
        {batchClassification && !isReadonly && (
	          <button
	            className={styles.sectionEditButton}
	            onClick={() => {
	              openEditModal<NonNullable<NonNullable<typeof batchClassification>['metrics']>>(
	                'Edit Batch Classification API Configuration',
	                batchMetrics || {},
	                [
	                  { name: 'enabled', label: 'Enable Metrics', type: 'boolean', description: 'Enable batch classification metrics collection' },
	                  { name: 'sample_rate', label: 'Sample Rate', type: 'percentage', placeholder: '10', description: 'Sampling rate for metrics collection (0-100%)', step: 1 },
	                  { name: 'detailed_goroutine_tracking', label: 'Detailed Goroutine Tracking', type: 'boolean', description: 'Track goroutine activity in batch classification metrics' },
	                  { name: 'high_resolution_timing', label: 'High Resolution Timing', type: 'boolean', description: 'Collect high-resolution latency metrics' },
	                  { name: 'batch_size_ranges', label: 'Batch Size Ranges (JSON)', type: 'json', placeholder: '[{"min":1,"max":8,"label":"1-8"}]' },
	                  { name: 'duration_buckets', label: 'Duration Buckets (JSON)', type: 'json', placeholder: '[0.01,0.05,0.1,0.5,1]' },
	                  { name: 'size_buckets', label: 'Size Buckets (JSON)', type: 'json', placeholder: '[1,8,16,32,64]' },
	                ],
	                async (data) => {
	                  const newConfig = cloneConfig(config)
	                  if (!newConfig.api) newConfig.api = {}
	                  newConfig.api.batch_classification = { metrics: data }
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
	        {batchClassification ? (
	          <>
	            <div className={styles.featureCard}>
	              <div className={styles.featureHeader}><span className={styles.featureTitle}>Batch Classification Metrics</span></div>
	              <div className={styles.featureBody}>
	                <div className={styles.configRow}><span className={styles.configLabel}>Metrics Enabled</span><span className={styles.configValue}>{batchMetrics?.enabled ? 'Yes' : 'No'}</span></div>
	                {batchMetrics?.sample_rate !== undefined && <div className={styles.configRow}><span className={styles.configLabel}>Sample Rate</span><span className={styles.configValue}>{((batchMetrics.sample_rate ?? 0) * 100).toFixed(0)}%</span></div>}
	                {batchMetrics?.batch_size_ranges?.length ? <div className={styles.configRow}><span className={styles.configLabel}>Batch Size Ranges</span><span className={styles.configValue}>{batchMetrics.batch_size_ranges.map((range) => range.label).join(', ')}</span></div> : null}
	                {batchMetrics?.duration_buckets?.length ? <div className={styles.configRow}><span className={styles.configLabel}>Duration Buckets</span><span className={styles.configValue}>{batchMetrics.duration_buckets.join(', ')}</span></div> : null}
	                {batchMetrics?.size_buckets?.length ? <div className={styles.configRow}><span className={styles.configLabel}>Size Buckets</span><span className={styles.configValue}>{batchMetrics.size_buckets.join(', ')}</span></div> : null}
	              </div>
	            </div>

	            {batchMetrics && (
	              <div className={styles.featureCard}>
	                <div className={styles.featureHeader}>
	                  <span className={styles.featureTitle}>Metrics Collection</span>
	                  <span className={`${styles.statusBadge} ${batchMetrics.enabled ? styles.statusActive : styles.statusInactive}`}>
	                    {batchMetrics.enabled ? '✓ Enabled' : '✗ Disabled'}
	                  </span>
	                </div>
	                {batchMetrics.enabled && (
	                  <div className={styles.featureBody}>
	                    {batchMetrics.sample_rate !== undefined && <div className={styles.configRow}><span className={styles.configLabel}>Sample Rate</span><span className={styles.configValue}>{((batchMetrics.sample_rate ?? 0) * 100).toFixed(0)}%</span></div>}
	                    {batchMetrics.detailed_goroutine_tracking !== undefined && <div className={styles.configRow}><span className={styles.configLabel}>Goroutine Tracking</span><span className={styles.configValue}>{batchMetrics.detailed_goroutine_tracking ? 'Yes' : 'No'}</span></div>}
	                    {batchMetrics.high_resolution_timing !== undefined && <div className={styles.configRow}><span className={styles.configLabel}>High Resolution Timing</span><span className={styles.configValue}>{batchMetrics.high_resolution_timing ? 'Yes' : 'No'}</span></div>}
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
