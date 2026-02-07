import { useCallback, useState } from 'react';
import type { EvaluationResult, TaskResults, TestCaseDetail, EvaluationMetadata } from '../../types/evaluation';
import { DIMENSION_INFO, LEVEL_INFO, formatDate, formatDuration, formatMetricValue, getMetricValue } from '../../types/evaluation';
import { downloadExport } from '../../utils/evaluationApi';
import styles from './ReportViewer.module.css';

interface ReportViewerProps {
  results: TaskResults;
  onBack?: () => void;
}

export function ReportViewer({ results, onBack }: ReportViewerProps) {
  const { task, results: evaluationResults } = results;

  const handleExport = useCallback(async (format: 'json' | 'csv') => {
    try {
      await downloadExport(task.id, format);
    } catch (err) {
      console.error('Export failed:', err);
    }
  }, [task.id]);

  const getOverallScore = useCallback(() => {
    // Calculate an overall score based on available metrics
    let totalScore = 0;
    let count = 0;

    for (const result of evaluationResults) {
      const accuracy = getMetricValue(result.metrics, 'accuracy');
      if (accuracy !== null) {
        totalScore += accuracy;
        count++;
      }
      const f1 = getMetricValue(result.metrics, 'f1_score');
      if (f1 !== null) {
        totalScore += f1;
        count++;
      }
    }

    return count > 0 ? totalScore / count : null;
  }, [evaluationResults]);

  const overallScore = getOverallScore();

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          {onBack && (
            <button className={styles.backButton} onClick={onBack}>
              Back
            </button>
          )}
          <div className={styles.taskInfo}>
            <h2>{task.name}</h2>
            {task.description && <p>{task.description}</p>}
          </div>
        </div>
        <div className={styles.headerRight}>
          <button className={styles.exportButton} onClick={() => handleExport('json')}>
            Export JSON
          </button>
          <button className={styles.exportButton} onClick={() => handleExport('csv')}>
            Export CSV
          </button>
        </div>
      </div>

      <div className={styles.summary}>
        <div className={styles.summaryCard}>
          <span className={styles.summaryLabel}>Status</span>
          <span className={styles.summaryValue} style={{ color: task.status === 'completed' ? '#22c55e' : '#ef4444' }}>
            {task.status === 'completed' ? 'Completed' : 'Failed'}
          </span>
        </div>
        <div className={styles.summaryCard}>
          <span className={styles.summaryLabel}>Duration</span>
          <span className={styles.summaryValue}>
            {formatDuration(task.started_at, task.completed_at)}
          </span>
        </div>
        <div className={styles.summaryCard}>
          <span className={styles.summaryLabel}>Dimensions</span>
          <span className={styles.summaryValue}>{task.config.dimensions.length}</span>
        </div>
        {overallScore !== null && (
          <div className={styles.summaryCard}>
            <span className={styles.summaryLabel}>Overall Score</span>
            <span className={styles.summaryValue} style={{ color: overallScore >= 0.8 ? '#22c55e' : overallScore >= 0.6 ? '#f59e0b' : '#ef4444' }}>
              {formatMetricValue(overallScore, 'percent')}
            </span>
          </div>
        )}
      </div>

      <div className={styles.results}>
        {evaluationResults.map((result) => (
          <ResultCard key={result.id} result={result} />
        ))}
      </div>

      <div className={styles.metadata}>
        <h3>Evaluation Details</h3>
        <dl className={styles.metadataList}>
          <dt>Task ID</dt>
          <dd>{task.id}</dd>
          <dt>Evaluation Level</dt>
          <dd>
            <span style={{ color: LEVEL_INFO[task.config.level].color }}>
              {LEVEL_INFO[task.config.level].label}
            </span>
          </dd>
          <dt>Created</dt>
          <dd>{formatDate(task.created_at)}</dd>
          <dt>Started</dt>
          <dd>{formatDate(task.started_at)}</dd>
          <dt>Completed</dt>
          <dd>{formatDate(task.completed_at)}</dd>
          <dt>Endpoint</dt>
          <dd>{task.config.endpoint}</dd>
          <dt>Max Samples</dt>
          <dd>{task.config.max_samples}</dd>
        </dl>
      </div>
    </div>
  );
}

interface ResultCardProps {
  result: EvaluationResult;
}

function ResultCard({ result }: ResultCardProps) {
  const dimInfo = DIMENSION_INFO[result.dimension];
  const [showDetails, setShowDetails] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  // Extract common metrics
  const accuracy = getMetricValue(result.metrics, 'accuracy');
  const precision = getMetricValue(result.metrics, 'precision');
  const recall = getMetricValue(result.metrics, 'recall');
  const f1 = getMetricValue(result.metrics, 'f1_score');
  const avgLatency = getMetricValue(result.metrics, 'avg_latency_ms');
  const p50Latency = getMetricValue(result.metrics, 'p50_latency_ms');
  const p99Latency = getMetricValue(result.metrics, 'p99_latency_ms');

  // Extract metadata and details
  const metadata = result.metrics.metadata as EvaluationMetadata | undefined;
  const details = result.metrics.details as TestCaseDetail[] | undefined;

  return (
    <div className={styles.resultCard}>
      <div className={styles.resultHeader}>
        <div className={styles.resultTitle}>
          <span className={styles.dimensionIndicator} style={{ backgroundColor: dimInfo?.color }} />
          <span className={styles.dimensionLabel}>{dimInfo?.label || result.dimension}</span>
        </div>
        <span className={styles.datasetName}>{result.dataset_name}</span>
      </div>

      {/* Dimension Info Box */}
      {((metadata?.description || dimInfo?.description) || metadata) && (
        <div className={styles.dimensionInfoBox}>
          {(metadata?.description || dimInfo?.description) && (
            <div className={styles.infoItem}>
              <span className={styles.infoIcon}>📋</span>
              <div className={styles.infoText}>
                <div>Dimension: {dimInfo?.label || result.dimension}</div>
                <span className={styles.datasetId}>{metadata?.description || dimInfo?.description}</span>
              </div>
            </div>
          )}
          {metadata && (
            <div className={styles.infoItem}>
              <span className={styles.infoIcon}>📊</span>
              <div className={styles.infoText}>
                <div>Dataset: {metadata.dataset_name || result.dataset_name}</div>
                {metadata.hf_dataset && <span className={styles.datasetId}>{metadata.hf_dataset}</span>}
              </div>
            </div>
          )}
        </div>
      )}

      <div className={styles.metricsGrid}>
        {accuracy !== null && (
          <div className={styles.metric}>
            <span className={styles.metricLabel}>Accuracy</span>
            <span className={styles.metricValue}>{formatMetricValue(accuracy, 'percent')}</span>
          </div>
        )}
        {precision !== null && (
          <div className={styles.metric}>
            <span className={styles.metricLabel}>Precision</span>
            <span className={styles.metricValue}>{formatMetricValue(precision, 'percent')}</span>
          </div>
        )}
        {recall !== null && (
          <div className={styles.metric}>
            <span className={styles.metricLabel}>Recall</span>
            <span className={styles.metricValue}>{formatMetricValue(recall, 'percent')}</span>
          </div>
        )}
        {f1 !== null && (
          <div className={styles.metric}>
            <span className={styles.metricLabel}>F1 Score</span>
            <span className={styles.metricValue}>{formatMetricValue(f1, 'percent')}</span>
          </div>
        )}
        {avgLatency !== null && (
          <div className={styles.metric}>
            <span className={styles.metricLabel}>Avg Latency</span>
            <span className={styles.metricValue}>{formatMetricValue(avgLatency, 'ms')}</span>
          </div>
        )}
        {p50Latency !== null && (
          <div className={styles.metric}>
            <span className={styles.metricLabel}>P50 Latency</span>
            <span className={styles.metricValue}>{formatMetricValue(p50Latency, 'ms')}</span>
          </div>
        )}
        {p99Latency !== null && (
          <div className={styles.metric}>
            <span className={styles.metricLabel}>P99 Latency</span>
            <span className={styles.metricValue}>{formatMetricValue(p99Latency, 'ms')}</span>
          </div>
        )}
      </div>

      {(result.dimension === 'domain' || result.dimension === 'fact_check' || result.dimension === 'user_feedback') && (
        <SignalEvalDetails metrics={result.metrics} />
      )}

      {/* Test Case Details */}
      {details && details.length > 0 && (() => {
        const totalPages = Math.ceil(details.length / itemsPerPage);
        const startIdx = (currentPage - 1) * itemsPerPage;
        const endIdx = startIdx + itemsPerPage;
        const currentItems = details.slice(startIdx, endIdx);

        return (
          <div className={styles.testCaseSection}>
            <button
              className={styles.toggleDetailsButton}
              onClick={() => setShowDetails(!showDetails)}
            >
              {showDetails ? '▼' : '▶'} Test Cases ({details.length})
            </button>
            {showDetails && (
              <>
                <div className={styles.testCaseList}>
                  {currentItems.map((testCase, idx) => {
                    const actualIdx = startIdx + idx;
                    return (
                      <div key={actualIdx} className={`${styles.testCase} ${styles[`testCase${testCase.status.charAt(0).toUpperCase() + testCase.status.slice(1)}`]}`}>
                        <div className={styles.testCaseHeader}>
                          <span className={styles.testCaseIndex}>#{actualIdx + 1}</span>
                          <span className={`${styles.testCaseStatus} ${styles[`status${testCase.status.charAt(0).toUpperCase() + testCase.status.slice(1)}`]}`}>
                            {testCase.status === 'correct' ? '✓' : testCase.status === 'incorrect' ? '✗' : '⊘'}
                          </span>
                        </div>
                        <div className={styles.testCaseContent}>
                          <div className={styles.testCaseQuery}>
                            <strong>Query:</strong> {testCase.query}
                          </div>
                          <div className={styles.testCaseComparison}>
                            <div className={styles.testCaseExpected}>
                              <strong>Expected:</strong> <code>{String(testCase.expected)}</code>
                            </div>
                            <div className={styles.testCaseActual}>
                              <strong>Actual:</strong> <code>{testCase.actual !== null ? String(testCase.actual) : 'N/A'}</code>
                            </div>
                          </div>
                          {testCase.reason && (
                            <div className={styles.testCaseReason}>
                              <strong>Reason:</strong> {testCase.reason}
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
                {totalPages > 1 && (
                  <div className={styles.pagination}>
                    <span className={styles.paginationInfo}>
                      Page {currentPage} of {totalPages} ({startIdx + 1}-{Math.min(endIdx, details.length)} of {details.length})
                    </span>
                    <div className={styles.paginationButtons}>
                      <button
                        className={styles.paginationButton}
                        onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                        disabled={currentPage === 1}
                      >
                        ← Previous
                      </button>
                      <button
                        className={styles.paginationButton}
                        onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                        disabled={currentPage === totalPages}
                      >
                        Next →
                      </button>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        );
      })()}
    </div>
  );
}

function SignalEvalDetails({ metrics }: { metrics: Record<string, unknown> }) {
  const totalSamples = getMetricValue(metrics, 'total_samples');
  const correct = getMetricValue(metrics, 'correct');
  const incorrect = getMetricValue(metrics, 'incorrect');
  const skipped = getMetricValue(metrics, 'skipped');

  return (
    <div className={styles.details}>
      <h4>Signal Evaluation Results</h4>
      <div className={styles.comparisonGrid}>
        <div className={styles.comparisonItem}>
          <span className={styles.comparisonLabel}>Total Samples</span>
          <span className={styles.comparisonValue}>{totalSamples ?? '-'}</span>
        </div>
        <div className={styles.comparisonItem}>
          <span className={styles.comparisonLabel}>Correct</span>
          <span className={styles.comparisonValue} style={{ color: '#22c55e' }}>
            {correct ?? '-'}
          </span>
        </div>
        <div className={styles.comparisonItem}>
          <span className={styles.comparisonLabel}>Incorrect</span>
          <span className={styles.comparisonValue} style={{ color: '#ef4444' }}>
            {incorrect ?? '-'}
          </span>
        </div>
        <div className={styles.comparisonItem}>
          <span className={styles.comparisonLabel}>Skipped</span>
          <span className={styles.comparisonValue} style={{ color: '#f59e0b' }}>
            {skipped ?? '-'}
          </span>
        </div>
      </div>
    </div>
  );
}

export default ReportViewer;
