// Evaluation system TypeScript types

export type EvaluationStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export type EvaluationLevel = 'router' | 'mom';

export type EvaluationDimension =
  | 'domain'
  | 'fact_check'
  | 'user_feedback';

export interface EvaluationConfig {
  level: EvaluationLevel; // evaluation level (router or mom)
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

export interface TestCaseDetail {
  query: string;
  expected: string | number;
  actual: string | number | null;
  status: 'correct' | 'incorrect' | 'skip';
  reason?: string;
}

export interface EvaluationMetadata {
  dataset_id: string;
  dataset_name: string;
  description: string;
  hf_dataset: string;
  dimension: string;
  endpoint: string;
  max_samples?: number;
  concurrent?: number;
  elapsed_time_seconds?: number;
  timestamp?: string;
}

export interface EvaluationResult {
  id: string;
  task_id: string;
  dimension: EvaluationDimension;
  dataset_name: string;
  metrics: Record<string, unknown> & {
    details?: TestCaseDetail[];
    metadata?: EvaluationMetadata;
    correct?: number;
    incorrect?: number;
    skipped?: number;
    accuracy?: number;
  };
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
  level: EvaluationLevel; // evaluation level (router or mom)
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

// Level metadata for UI display
export const LEVEL_INFO: Record<EvaluationLevel, { label: string; description: string; color: string }> = {
  router: {
    label: 'Signal Level',
    description: 'Evaluates the signal extraction accuracy (domain, fact_check, user_feedback)',
    color: '#10b981', // green
  },
  mom: {
    label: 'System Level',
    description: 'Evaluates the system as a unified model (reasoning, coding, agentic)',
    color: '#3b82f6', // blue
  },
};

// Dimension metadata for UI display
export const DIMENSION_INFO: Record<EvaluationDimension, { label: string; description: string; color: string }> = {
  domain: {
    label: 'Domain Classification',
    description: 'Evaluates intent signal extraction accuracy',
    color: '#8b5cf6', // purple
  },
  fact_check: {
    label: 'Fact Check Detection',
    description: 'Evaluates fact-check signal extraction accuracy',
    color: '#ef4444', // red
  },
  user_feedback: {
    label: 'User Feedback Detection',
    description: 'Evaluates feedback signal extraction accuracy',
    color: '#22c55e', // green
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
