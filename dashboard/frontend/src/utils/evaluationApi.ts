// Evaluation API client utilities

import type {
  EvaluationTask,
  CreateTaskRequest,
  RunTaskRequest,
  DatasetInfo,
  ProgressUpdate,
  TaskResults,
  EvaluationHistoryEntry,
} from '../types/evaluation';

const API_BASE = '/api/evaluation';

// Error handling helper
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}: ${response.statusText}`);
  }
  return response.json();
}

// List all evaluation tasks
export async function listTasks(status?: string): Promise<EvaluationTask[]> {
  const url = status ? `${API_BASE}/tasks?status=${status}` : `${API_BASE}/tasks`;
  const response = await fetch(url);
  return handleResponse<EvaluationTask[]>(response);
}

// Get a specific task by ID
export async function getTask(taskId: string): Promise<EvaluationTask> {
  const response = await fetch(`${API_BASE}/tasks/${taskId}`);
  return handleResponse<EvaluationTask>(response);
}

// Create a new evaluation task
export async function createTask(request: CreateTaskRequest): Promise<EvaluationTask> {
  const response = await fetch(`${API_BASE}/tasks`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  return handleResponse<EvaluationTask>(response);
}

// Delete a task
export async function deleteTask(taskId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/tasks/${taskId}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}: ${response.statusText}`);
  }
}

// Run an evaluation task
export async function runTask(request: RunTaskRequest): Promise<{ status: string; task_id: string }> {
  const response = await fetch(`${API_BASE}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  return handleResponse<{ status: string; task_id: string }>(response);
}

// Cancel a running task
export async function cancelTask(taskId: string): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE}/cancel/${taskId}`, {
    method: 'POST',
  });
  return handleResponse<{ status: string }>(response);
}

// Get results for a completed task
export async function getResults(taskId: string): Promise<TaskResults> {
  const response = await fetch(`${API_BASE}/results/${taskId}`);
  return handleResponse<TaskResults>(response);
}

// Export results in specified format
export async function exportResults(taskId: string, format: 'json' | 'csv' = 'json'): Promise<Blob> {
  const response = await fetch(`${API_BASE}/export/${taskId}?format=${format}`);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}: ${response.statusText}`);
  }
  return response.blob();
}

// Get available datasets grouped by dimension
export async function getDatasets(): Promise<Record<string, DatasetInfo[]>> {
  const response = await fetch(`${API_BASE}/datasets`);
  return handleResponse<Record<string, DatasetInfo[]>>(response);
}

// Get historical metric entries
export async function getHistory(metricName: string, limit: number = 100): Promise<EvaluationHistoryEntry[]> {
  const response = await fetch(`${API_BASE}/history?metric=${metricName}&limit=${limit}`);
  return handleResponse<EvaluationHistoryEntry[]>(response);
}

// Subscribe to progress updates via SSE
export function subscribeToProgress(
  taskId: string,
  onProgress: (update: ProgressUpdate) => void,
  onComplete: () => void,
  onError: (error: Error) => void
): () => void {
  const eventSource = new EventSource(`${API_BASE}/stream/${taskId}`);

  eventSource.addEventListener('connected', () => {
    console.log(`Connected to progress stream for task ${taskId}`);
  });

  eventSource.addEventListener('progress', (event) => {
    try {
      const update = JSON.parse(event.data) as ProgressUpdate;
      onProgress(update);
    } catch (err) {
      console.error('Failed to parse progress update:', err);
    }
  });

  eventSource.addEventListener('completed', () => {
    eventSource.close();
    onComplete();
  });

  eventSource.onerror = (event) => {
    console.error('SSE error:', event);
    eventSource.close();
    onError(new Error('Connection to progress stream lost'));
  };

  // Return cleanup function
  return () => {
    eventSource.close();
  };
}

// Download exported results as a file
export async function downloadExport(taskId: string, format: 'json' | 'csv' = 'json'): Promise<void> {
  const blob = await exportResults(taskId, format);
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `evaluation_${taskId.slice(0, 8)}.${format}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// Utility to create default config
export function createDefaultConfig(): CreateTaskRequest['config'] {
  return {
    dimensions: ['hallucination'],
    datasets: { hallucination: ['halueval'] },
    max_samples: 50,
    endpoint: 'http://localhost:8801',
    model: 'MoM',
    concurrent: 1,
    samples_per_cat: 10,
  };
}
