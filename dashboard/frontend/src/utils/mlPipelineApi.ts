// ML Pipeline API client utilities

import type {
  MLJob,
  MLProgressUpdate,
  BenchmarkConfig,
  TrainConfig,
  ConfigGenerationRequest,
} from '../types/mlPipeline';

const API_BASE = '/api/ml-pipeline';

// Error handling helper
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}: ${response.statusText}`);
  }
  return response.json();
}

// List all ML pipeline jobs
export async function listJobs(): Promise<MLJob[]> {
  const response = await fetch(`${API_BASE}/jobs`);
  return handleResponse<MLJob[]>(response);
}

// Get a specific job by ID
export async function getJob(jobId: string): Promise<MLJob> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}`);
  return handleResponse<MLJob>(response);
}

// Start a benchmark job (Layer 1)
export async function startBenchmark(
  modelsYaml: File,
  queriesJsonl: File,
  config: BenchmarkConfig
): Promise<{ job_id: string; status: string }> {
  const formData = new FormData();
  formData.append('models_yaml', modelsYaml);
  formData.append('queries_jsonl', queriesJsonl);
  formData.append('config', JSON.stringify(config));

  const response = await fetch(`${API_BASE}/benchmark`, {
    method: 'POST',
    body: formData,
  });
  return handleResponse<{ job_id: string; status: string }>(response);
}

// Start a training job from a previous benchmark job (Layer 2)
export async function startTraining(
  benchmarkJobId: string,
  config: TrainConfig
): Promise<{ job_id: string; status: string }> {
  const response = await fetch(`${API_BASE}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      benchmark_job_id: benchmarkJobId,
      config,
    }),
  });
  return handleResponse<{ job_id: string; status: string }>(response);
}

// Start a training job with an uploaded training data file (skip benchmark)
export async function startTrainingWithFile(
  trainingDataFile: File,
  config: TrainConfig
): Promise<{ job_id: string; status: string }> {
  const formData = new FormData();
  formData.append('training_data', trainingDataFile);
  formData.append('config', JSON.stringify(config));

  const response = await fetch(`${API_BASE}/train`, {
    method: 'POST',
    body: formData,
  });
  return handleResponse<{ job_id: string; status: string }>(response);
}

// Generate deployment config (Layer 3)
export async function generateConfig(
  request: ConfigGenerationRequest
): Promise<{ job_id: string; status: string }> {
  const response = await fetch(`${API_BASE}/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  return handleResponse<{ job_id: string; status: string }>(response);
}

// Download a job's output file
export function getDownloadUrl(jobId: string, fileIndex = 0): string {
  return `${API_BASE}/download/${jobId}/${fileIndex}`;
}

// Subscribe to job progress updates via SSE with auto-reconnect.
// The Go backend sends heartbeats every 15s and replays last-known progress
// on connect, so reconnection is seamless.
export function subscribeToProgress(
  jobId: string,
  onProgress: (update: MLProgressUpdate) => void,
  onComplete: () => void,
  onError: (error: Error) => void
): () => void {
  let cancelled = false;
  let retryCount = 0;
  const MAX_RETRIES = 10;
  let currentSource: EventSource | null = null;

  function connect() {
    if (cancelled) return;

    const eventSource = new EventSource(`${API_BASE}/stream/${jobId}`);
    currentSource = eventSource;

    eventSource.addEventListener('connected', () => {
      console.log(`Connected to ML pipeline progress stream for job ${jobId}`);
      retryCount = 0; // reset on successful connect
    });

    eventSource.addEventListener('progress', (event) => {
      try {
        const update = JSON.parse(event.data) as MLProgressUpdate;
        onProgress(update);
      } catch (err) {
        console.error('Failed to parse ML progress update:', err);
      }
    });

    eventSource.addEventListener('completed', () => {
      eventSource.close();
      currentSource = null;
      onComplete();
    });

    eventSource.onerror = () => {
      eventSource.close();
      currentSource = null;

      if (cancelled) return;

      retryCount++;
      if (retryCount > MAX_RETRIES) {
        console.error(`ML pipeline SSE: giving up after ${MAX_RETRIES} retries`);
        onError(new Error('Connection to ML progress stream lost after multiple retries'));
        return;
      }

      // Exponential backoff: 1s, 2s, 4s, 8s... capped at 15s
      const delay = Math.min(1000 * Math.pow(2, retryCount - 1), 15000);
      console.log(`ML pipeline SSE: reconnecting in ${delay}ms (attempt ${retryCount}/${MAX_RETRIES})`);
      setTimeout(connect, delay);
    };
  }

  connect();

  return () => {
    cancelled = true;
    if (currentSource) {
      currentSource.close();
      currentSource = null;
    }
  };
}
