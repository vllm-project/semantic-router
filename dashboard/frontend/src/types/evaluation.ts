// Evaluation system TypeScript types

export type EvaluationStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export type EvaluationDimension =
  | 'hallucination'
  | 'reasoning'
  | 'accuracy'
  | 'latency'
  | 'cost'
  | 'security';

export interface EvaluationConfig {
  dimensions: EvaluationDimension[];
  datasets: Record<string, string[]>; // dimension -> dataset names
  max_samples: number;
  endpoint: string;
  model: string;
  concurrent: number;
  samples_per_cat: number;
}

export interface EvaluationTask {
  id: string;
  name: string;
  description: string;
  status: EvaluationStatus;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  config: EvaluationConfig;
  error_message?: string;
  progress_percent: number;
  current_step?: string;
}

export interface EvaluationResult {
  id: string;
  task_id: string;
  dimension: EvaluationDimension;
  dataset_name: string;
  metrics: Record<string, unknown>;
  raw_results_path?: string;
}

export interface EvaluationHistoryEntry {
  id: number;
  result_id: string;
  metric_name: string;
  metric_value: number;
  recorded_at: string;
}

export interface DatasetInfo {
  name: string;
  description: string;
  dimension: EvaluationDimension;
  sample_count?: number;
}

export interface CreateTaskRequest {
  name: string;
  description: string;
  config: EvaluationConfig;
}

export interface RunTaskRequest {
  task_id: string;
}

export interface ProgressUpdate {
  task_id: string;
  progress_percent: number;
  current_step: string;
  message?: string;
  timestamp: number;
}

export interface TaskResults {
  task: EvaluationTask;
  results: EvaluationResult[];
}

// Dimension metadata for UI display
export const DIMENSION_INFO: Record<EvaluationDimension, { label: string; description: string; color: string }> = {
  hallucination: {
    label: 'Hallucination Detection',
    description: 'Measures the system ability to detect factual inaccuracies',
    color: '#ef4444', // red
  },
  reasoning: {
    label: 'Reasoning Mode',
    description: 'Compares standard vs reasoning mode performance',
    color: '#8b5cf6', // purple
  },
  accuracy: {
    label: 'Classification Accuracy',
    description: 'Measures routing and classification correctness',
    color: '#22c55e', // green
  },
  latency: {
    label: 'Latency',
    description: 'Measures response time and throughput',
    color: '#3b82f6', // blue
  },
  cost: {
    label: 'Cost Efficiency',
    description: 'Measures token usage and cost per request',
    color: '#f59e0b', // amber
  },
  security: {
    label: 'Security',
    description: 'Measures jailbreak and prompt injection detection',
    color: '#ec4899', // pink
  },
};

// Status metadata for UI display
export const STATUS_INFO: Record<EvaluationStatus, { label: string; color: string; bgColor: string }> = {
  pending: {
    label: 'Pending',
    color: '#6b7280',
    bgColor: 'rgba(107, 114, 128, 0.15)',
  },
  running: {
    label: 'Running',
    color: '#3b82f6',
    bgColor: 'rgba(59, 130, 246, 0.15)',
  },
  completed: {
    label: 'Completed',
    color: '#22c55e',
    bgColor: 'rgba(34, 197, 94, 0.15)',
  },
  failed: {
    label: 'Failed',
    color: '#ef4444',
    bgColor: 'rgba(239, 68, 68, 0.15)',
  },
  cancelled: {
    label: 'Cancelled',
    color: '#f59e0b',
    bgColor: 'rgba(245, 158, 11, 0.15)',
  },
};

// Helper functions
export function formatDuration(startTime?: string, endTime?: string): string {
  if (!startTime) return '-';
  const start = new Date(startTime).getTime();
  const end = endTime ? new Date(endTime).getTime() : Date.now();
  const durationMs = end - start;

  if (durationMs < 1000) return `${durationMs}ms`;
  if (durationMs < 60000) return `${(durationMs / 1000).toFixed(1)}s`;
  if (durationMs < 3600000) return `${Math.floor(durationMs / 60000)}m ${Math.floor((durationMs % 60000) / 1000)}s`;
  return `${Math.floor(durationMs / 3600000)}h ${Math.floor((durationMs % 3600000) / 60000)}m`;
}

export function formatDate(dateString?: string): string {
  if (!dateString) return '-';
  return new Date(dateString).toLocaleString();
}

export function getMetricValue(metrics: Record<string, unknown>, key: string): number | null {
  const value = metrics[key];
  if (typeof value === 'number') return value;
  if (typeof value === 'string') {
    const parsed = parseFloat(value);
    return isNaN(parsed) ? null : parsed;
  }
  return null;
}

export function formatMetricValue(value: number | null, format: 'percent' | 'decimal' | 'ms' | 'count' = 'decimal'): string {
  if (value === null) return '-';
  switch (format) {
    case 'percent':
      return `${(value * 100).toFixed(1)}%`;
    case 'ms':
      return `${value.toFixed(1)}ms`;
    case 'count':
      return value.toFixed(0);
    default:
      return value.toFixed(3);
  }
}
