// ML Pipeline TypeScript types

export type MLJobStatus = 'pending' | 'running' | 'completed' | 'failed';

export type MLAlgorithm = 'knn' | 'kmeans' | 'svm' | 'mlp';

export type PipelineStep = 'benchmark' | 'train' | 'config';

export interface MLJob {
  id: string;
  type: PipelineStep;
  status: MLJobStatus;
  created_at: string;
  completed_at?: string;
  error?: string;
  output_files?: string[];
  progress: number;
  current_step: string;
}

export interface MLProgressUpdate {
  job_id: string;
  step: string;
  percent: number;
  message: string;
}

export interface BenchmarkConfig {
  concurrency: number;
  max_tokens?: number;      // default 1024
  temperature?: number;     // default 0.0
  concise?: boolean;        // use concise prompts for faster inference
  limit?: number;           // limit number of queries (0 = no limit)
}

export type SvmKernel = 'rbf' | 'linear';

export interface TrainConfig {
  algorithms: MLAlgorithm[];
  device: 'cpu' | 'cuda' | 'mps';
  quality_weight?: number;    // default 0.9
  batch_size?: number;        // default 32
  // KNN
  knn_k?: number;             // default 5
  // KMeans
  kmeans_clusters?: number;   // default 8
  // SVM
  svm_kernel?: SvmKernel;     // default "rbf"
  svm_gamma?: number;         // default 1.0
  // MLP
  mlp_hidden_sizes?: string;  // e.g. "256,128"
  mlp_epochs?: number;        // default 100
  mlp_learning_rate?: number; // default 0.001
  mlp_dropout?: number;       // default 0.1
}

export interface DecisionEntry {
  name: string;
  domains: string[];
  algorithm: MLAlgorithm;
  priority: number;
  model_names: string[];
}

export interface ConfigGenerationRequest {
  models_path: string;
  device: 'cpu' | 'cuda' | 'mps';
  decisions: DecisionEntry[];
}

// Wizard state for the 3-step pipeline
export interface MLPipelineState {
  currentStep: number; // 0=benchmark, 1=train, 2=config
  benchmarkJobId: string | null;
  trainJobId: string | null;
  configJobId: string | null;
  benchmarkConfig: BenchmarkConfig;
  trainConfig: TrainConfig;
  configRequest: ConfigGenerationRequest;
}

// Status metadata for UI display
export const ML_JOB_STATUS_INFO: Record<MLJobStatus, { label: string; color: string; bgColor: string }> = {
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
};

export const ML_ALGORITHM_INFO: Record<MLAlgorithm, { label: string; description: string; color: string }> = {
  knn: {
    label: 'K-Nearest Neighbors',
    description: 'Distance-based classification using nearest training examples',
    color: '#8b5cf6',
  },
  kmeans: {
    label: 'K-Means Clustering',
    description: 'Centroid-based clustering for grouping similar queries',
    color: '#06b6d4',
  },
  svm: {
    label: 'Support Vector Machine',
    description: 'Margin-based classification with kernel functions',
    color: '#f59e0b',
  },
  mlp: {
    label: 'Multi-Layer Perceptron',
    description: 'Neural network classifier with hidden layers',
    color: '#ec4899',
  },
};

export const PIPELINE_STEPS: { key: PipelineStep; label: string; description: string }[] = [
  {
    key: 'benchmark',
    label: 'Benchmark',
    description: 'Run benchmark against your models to collect embedding data',
  },
  {
    key: 'train',
    label: 'Train',
    description: 'Train ML classification models on the benchmark data',
  },
  {
    key: 'config',
    label: 'Configure',
    description: 'Generate deployment-ready configuration for semantic-router',
  },
];
