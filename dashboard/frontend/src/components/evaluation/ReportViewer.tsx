import { useCallback } from 'react';
import type { EvaluationResult, TaskResults } from '../../types/evaluation';
import { DIMENSION_INFO, formatDate, formatDuration, formatMetricValue, getMetricValue } from '../../types/evaluation';
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

  // Extract common metrics
  const accuracy = getMetricValue(result.metrics, 'accuracy');
  const precision = getMetricValue(result.metrics, 'precision');
  const recall = getMetricValue(result.metrics, 'recall');
  const f1 = getMetricValue(result.metrics, 'f1_score');
  const avgLatency = getMetricValue(result.metrics, 'avg_latency_ms');
  const p50Latency = getMetricValue(result.metrics, 'p50_latency_ms');
  const p99Latency = getMetricValue(result.metrics, 'p99_latency_ms');

  return (
    <div className={styles.resultCard}>
      <div className={styles.resultHeader}>
        <div className={styles.resultTitle}>
          <span className={styles.dimensionIndicator} style={{ backgroundColor: dimInfo?.color }} />
          <span className={styles.dimensionLabel}>{dimInfo?.label || result.dimension}</span>
        </div>
        <span className={styles.datasetName}>{result.dataset_name}</span>
      </div>

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

      {result.dimension === 'reasoning' && (
        <ReasoningDetails metrics={result.metrics} />
      )}

      {result.dimension === 'hallucination' && (
        <HallucinationDetails metrics={result.metrics} />
      )}
    </div>
  );
}

function ReasoningDetails({ metrics }: { metrics: Record<string, unknown> }) {
  const standardAccuracy = getMetricValue(metrics, 'standard_accuracy');
  const reasoningAccuracy = getMetricValue(metrics, 'reasoning_accuracy');
  const accuracyDelta = getMetricValue(metrics, 'accuracy_delta');
  const improvementPct = getMetricValue(metrics, 'accuracy_improvement_pct');

  return (
    <div className={styles.details}>
      <h4>Reasoning Mode Comparison</h4>
      <div className={styles.comparisonGrid}>
        <div className={styles.comparisonItem}>
          <span className={styles.comparisonLabel}>Standard Mode</span>
          <span className={styles.comparisonValue}>
            {formatMetricValue(standardAccuracy, 'percent')}
          </span>
        </div>
        <div className={styles.comparisonItem}>
          <span className={styles.comparisonLabel}>Reasoning Mode</span>
          <span className={styles.comparisonValue} style={{ color: '#22c55e' }}>
            {formatMetricValue(reasoningAccuracy, 'percent')}
          </span>
        </div>
        {accuracyDelta !== null && (
          <div className={styles.comparisonItem}>
            <span className={styles.comparisonLabel}>Improvement</span>
            <span
              className={styles.comparisonValue}
              style={{ color: accuracyDelta >= 0 ? '#22c55e' : '#ef4444' }}
            >
              {accuracyDelta >= 0 ? '+' : ''}{formatMetricValue(improvementPct, 'decimal')}%
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function HallucinationDetails({ metrics }: { metrics: Record<string, unknown> }) {
  const tp = getMetricValue(metrics, 'true_positives');
  const fp = getMetricValue(metrics, 'false_positives');
  const tn = getMetricValue(metrics, 'true_negatives');
  const fn = getMetricValue(metrics, 'false_negatives');
  const efficiencyGain = getMetricValue(metrics, 'efficiency_gain_percent');

  return (
    <div className={styles.details}>
      <h4>Detection Results</h4>
      <div className={styles.confusionMatrix}>
        <div className={styles.matrixRow}>
          <div className={styles.matrixCell} style={{ backgroundColor: 'rgba(34, 197, 94, 0.2)' }}>
            <span className={styles.matrixLabel}>True Positives</span>
            <span className={styles.matrixValue}>{tp ?? '-'}</span>
          </div>
          <div className={styles.matrixCell} style={{ backgroundColor: 'rgba(239, 68, 68, 0.2)' }}>
            <span className={styles.matrixLabel}>False Positives</span>
            <span className={styles.matrixValue}>{fp ?? '-'}</span>
          </div>
        </div>
        <div className={styles.matrixRow}>
          <div className={styles.matrixCell} style={{ backgroundColor: 'rgba(239, 68, 68, 0.2)' }}>
            <span className={styles.matrixLabel}>False Negatives</span>
            <span className={styles.matrixValue}>{fn ?? '-'}</span>
          </div>
          <div className={styles.matrixCell} style={{ backgroundColor: 'rgba(34, 197, 94, 0.2)' }}>
            <span className={styles.matrixLabel}>True Negatives</span>
            <span className={styles.matrixValue}>{tn ?? '-'}</span>
          </div>
        </div>
      </div>
      {efficiencyGain !== null && (
        <div className={styles.efficiencyBanner}>
          <span className={styles.efficiencyLabel}>Efficiency Gain from Pre-filtering</span>
          <span className={styles.efficiencyValue}>{formatMetricValue(efficiencyGain, 'decimal')}%</span>
        </div>
      )}
    </div>
  );
}

export default ReportViewer;
