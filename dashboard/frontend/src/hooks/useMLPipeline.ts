// Custom hooks for ML pipeline

import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  MLJob,
  MLProgressUpdate,
  MLAlgorithm,
  BenchmarkConfig,
  TrainConfig,
  ConfigGenerationRequest,
  DecisionEntry,
  SvmKernel,
} from '../types/mlPipeline';
import * as api from '../utils/mlPipelineApi';

// Hook for managing the list of ML pipeline jobs
export function useMLJobs(autoRefresh = false, refreshInterval = 5000) {
  const [jobs, setJobs] = useState<MLJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchJobs = useCallback(async () => {
    try {
      const data = await api.listJobs();
      setJobs(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch ML jobs');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchJobs();
    if (autoRefresh) {
      const interval = setInterval(fetchJobs, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [fetchJobs, autoRefresh, refreshInterval]);

  const refresh = useCallback(() => {
    setLoading(true);
    fetchJobs();
  }, [fetchJobs]);

  return { jobs, loading, error, refresh };
}

// Hook for tracking a single job with SSE progress + polling fallback
export function useMLJobProgress(jobId: string | null) {
  const [job, setJob] = useState<MLJob | null>(null);
  const [progress, setProgress] = useState<MLProgressUpdate | null>(null);
  const [completed, setCompleted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);
  const sseActiveRef = useRef(false);

  // Fetch initial job state
  useEffect(() => {
    if (!jobId) {
      setJob(null);
      setProgress(null);
      setCompleted(false);
      return;
    }
    api.getJob(jobId).then(setJob).catch((err) => {
      setError(err instanceof Error ? err.message : 'Failed to fetch job');
    });
  }, [jobId]);

  // Subscribe to SSE progress
  useEffect(() => {
    if (!jobId) return;

    sseActiveRef.current = false;

    const cleanup = api.subscribeToProgress(
      jobId,
      (update) => {
        sseActiveRef.current = true;
        setProgress(update);
        setError(null);
        // Update job progress locally
        setJob((prev) =>
          prev ? { ...prev, progress: update.percent, current_step: update.step } : prev
        );
      },
      () => {
        setCompleted(true);
        // Fetch final job state
        api.getJob(jobId).then(setJob).catch(() => {});
      },
      (err) => {
        setError(err.message);
      }
    );

    cleanupRef.current = cleanup;
    return () => {
      cleanup();
      cleanupRef.current = null;
    };
  }, [jobId]);

  // Polling fallback: periodically fetch job state in case SSE is not delivering
  useEffect(() => {
    if (!jobId || completed) return;

    const poll = setInterval(() => {
      api.getJob(jobId).then((fetchedJob) => {
        setJob(fetchedJob);
        // If SSE hasn't delivered any events yet, use polled data to update progress
        if (!sseActiveRef.current && fetchedJob.progress > 0) {
          setProgress({
            job_id: jobId,
            step: fetchedJob.current_step || 'Processing...',
            percent: fetchedJob.progress,
            message: '',
          });
        }
        // Detect completion via polling if SSE missed the completed event
        if (fetchedJob.status === 'completed' || fetchedJob.status === 'failed') {
          setCompleted(true);
          if (fetchedJob.status === 'failed') {
            setError(fetchedJob.error || 'Job failed');
          }
        }
      }).catch(() => { /* ignore polling errors */ });
    }, 5000);

    return () => clearInterval(poll);
  }, [jobId, completed]);

  const disconnect = useCallback(() => {
    if (cleanupRef.current) {
      cleanupRef.current();
      cleanupRef.current = null;
    }
  }, []);

  return { job, progress, completed, error, disconnect };
}

// Hook for managing the 3-step ML onboarding wizard
export function useMLPipelineWizard() {
  const [currentStep, setCurrentStep] = useState(0); // 0=benchmark, 1=train, 2=config

  // Benchmark state (Step 0)
  const [modelsFile, setModelsFile] = useState<File | null>(null);
  const [queriesFile, setQueriesFile] = useState<File | null>(null);
  const [benchmarkConfig, setBenchmarkConfig] = useState<BenchmarkConfig>({ concurrency: 1 });
  const [benchmarkJobId, setBenchmarkJobId] = useState<string | null>(null);

  // Training state (Step 1)
  const [trainingDataFile, setTrainingDataFile] = useState<File | null>(null); // uploaded training data (skip benchmark)
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<MLAlgorithm[]>([]);
  const [device, setDevice] = useState<'cpu' | 'cuda' | 'mps'>('cpu');
  const [trainJobId, setTrainJobId] = useState<string | null>(null);
  // Advanced training params
  const [qualityWeight, setQualityWeight] = useState(0.9);
  const [batchSize, setBatchSize] = useState(32);
  const [knnK, setKnnK] = useState(5);
  const [kmeansClusters, setKmeansClusters] = useState(8);
  const [svmKernel, setSvmKernel] = useState<SvmKernel>('rbf');
  const [svmGamma, setSvmGamma] = useState(1.0);
  const [mlpHiddenSizes, setMlpHiddenSizes] = useState('256,128');
  const [mlpEpochs, setMlpEpochs] = useState(100);
  const [mlpLearningRate, setMlpLearningRate] = useState(0.001);
  const [mlpDropout, setMlpDropout] = useState(0.1);

  // Config state (Step 2)
  const [modelsPath, setModelsPath] = useState('/data/ml-pipeline/ml-train');
  const [decisions, setDecisions] = useState<DecisionEntry[]>([
    {
      name: 'default-decision',
      domains: ['general'],
      algorithm: 'knn',
      priority: 100,
      model_names: [],
    },
  ]);
  const [configJobId, setConfigJobId] = useState<string | null>(null);

  // Job progress tracking
  const benchmarkProgress = useMLJobProgress(benchmarkJobId);
  const trainProgress = useMLJobProgress(trainJobId);
  const configProgress = useMLJobProgress(configJobId);

  // Auto-populate models path from training output when training completes
  useEffect(() => {
    if (trainProgress.job?.status === 'completed' && trainProgress.job.output_files?.length) {
      // Extract directory from the first output file path (e.g., ".../ml-train/knn_model.json" → ".../ml-train")
      const firstFile = trainProgress.job.output_files[0];
      const dir = firstFile.substring(0, firstFile.lastIndexOf('/'));
      if (dir) {
        setModelsPath(dir);
      }
    }
  }, [trainProgress.job?.status, trainProgress.job?.output_files]);

  // Actions
  const [actionLoading, setActionLoading] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const startBenchmark = useCallback(async () => {
    if (!modelsFile || !queriesFile) {
      setActionError('Please upload both models YAML and queries JSONL files');
      return;
    }
    setActionLoading(true);
    setActionError(null);
    try {
      const result = await api.startBenchmark(modelsFile, queriesFile, benchmarkConfig);
      setBenchmarkJobId(result.job_id);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to start benchmark');
    } finally {
      setActionLoading(false);
    }
  }, [modelsFile, queriesFile, benchmarkConfig]);

  const startTraining = useCallback(async () => {
    if (!benchmarkJobId && !trainingDataFile) {
      setActionError('Please either complete the benchmark step or upload a training data file');
      return;
    }
    setActionLoading(true);
    setActionError(null);
    try {
      const config: TrainConfig = {
        algorithms: selectedAlgorithms,
        device,
        quality_weight: qualityWeight,
        batch_size: batchSize,
        knn_k: knnK,
        kmeans_clusters: kmeansClusters,
        svm_kernel: svmKernel,
        svm_gamma: svmGamma,
        mlp_hidden_sizes: mlpHiddenSizes,
        mlp_epochs: mlpEpochs,
        mlp_learning_rate: mlpLearningRate,
        mlp_dropout: mlpDropout,
      };
      let result;
      if (trainingDataFile) {
        // Upload mode: skip benchmark, use uploaded file directly
        result = await api.startTrainingWithFile(trainingDataFile, config);
      } else {
        // Normal mode: use output from benchmark job
        result = await api.startTraining(benchmarkJobId!, config);
      }
      setTrainJobId(result.job_id);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to start training');
    } finally {
      setActionLoading(false);
    }
  }, [benchmarkJobId, trainingDataFile, selectedAlgorithms, device, qualityWeight, batchSize, knnK, kmeansClusters, svmKernel, svmGamma, mlpHiddenSizes, mlpEpochs, mlpLearningRate, mlpDropout]);

  const startConfigGeneration = useCallback(async () => {
    setActionLoading(true);
    setActionError(null);
    try {
      const request: ConfigGenerationRequest = {
        models_path: modelsPath,
        device,
        decisions,
      };
      const result = await api.generateConfig(request);
      setConfigJobId(result.job_id);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to generate config');
    } finally {
      setActionLoading(false);
    }
  }, [modelsPath, device, decisions]);

  const toggleAlgorithm = useCallback((alg: MLAlgorithm) => {
    setSelectedAlgorithms((prev) =>
      prev.includes(alg) ? prev.filter((a) => a !== alg) : [...prev, alg]
    );
  }, []);

  const addDecision = useCallback(() => {
    setDecisions((prev) => [
      ...prev,
      {
        name: `decision-${prev.length + 1}`,
        domains: [],
        algorithm: 'knn',
        priority: 100,
        model_names: [],
      },
    ]);
  }, []);

  const updateDecision = useCallback((index: number, updates: Partial<DecisionEntry>) => {
    setDecisions((prev) => prev.map((d, i) => (i === index ? { ...d, ...updates } : d)));
  }, []);

  const removeDecision = useCallback((index: number) => {
    setDecisions((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const nextStep = useCallback(() => setCurrentStep((s) => Math.min(s + 1, 2)), []);
  const prevStep = useCallback(() => setCurrentStep((s) => Math.max(s - 1, 0)), []);
  const goToStep = useCallback((step: number) => setCurrentStep(step), []);

  // Whether benchmark was skipped (user uploaded training data directly)
  const benchmarkSkipped = trainingDataFile !== null;

  const canAdvanceToStep = useCallback(
    (step: number): boolean => {
      switch (step) {
        case 1:
          // Can go to train if benchmark completed OR user uploaded training data
          return benchmarkSkipped || (benchmarkProgress.completed && benchmarkProgress.job?.status === 'completed');
        case 2:
          return trainProgress.completed && trainProgress.job?.status === 'completed';
        default:
          return true;
      }
    },
    [benchmarkSkipped, benchmarkProgress.completed, benchmarkProgress.job?.status, trainProgress.completed, trainProgress.job?.status]
  );

  const clearError = useCallback(() => setActionError(null), []);

  // Reset training only — keeps benchmark data, allows re-training with different algorithms
  const resetTraining = useCallback(() => {
    setTrainJobId(null);
    setSelectedAlgorithms([]);
    setDevice('cpu');
    setQualityWeight(0.9);
    setBatchSize(32);
    setKnnK(5);
    setKmeansClusters(8);
    setSvmKernel('rbf');
    setSvmGamma(1.0);
    setMlpHiddenSizes('256,128');
    setMlpEpochs(100);
    setMlpLearningRate(0.001);
    setMlpDropout(0.1);
    setActionError(null);
  }, []);

  const reset = useCallback(() => {
    setCurrentStep(0);
    setModelsFile(null);
    setQueriesFile(null);
    setBenchmarkConfig({ concurrency: 1 });
    setBenchmarkJobId(null);
    setTrainingDataFile(null);
    setSelectedAlgorithms([]);
    setDevice('cpu');
    setTrainJobId(null);
    setModelsPath('/data/ml-pipeline/ml-train');
    setDecisions([
      {
        name: 'default-decision',
        domains: ['general'],
        algorithm: 'knn',
        priority: 100,
        model_names: [],
      },
    ]);
    setConfigJobId(null);
    setActionError(null);
  }, []);

  return {
    // Step navigation
    currentStep,
    nextStep,
    prevStep,
    goToStep,
    canAdvanceToStep,

    // Benchmark (Step 0)
    modelsFile,
    setModelsFile,
    queriesFile,
    setQueriesFile,
    benchmarkConfig,
    setBenchmarkConfig,
    benchmarkJobId,
    benchmarkProgress,
    startBenchmark,

    // Training (Step 1)
    trainingDataFile,
    setTrainingDataFile,
    benchmarkSkipped,
    selectedAlgorithms,
    toggleAlgorithm,
    device,
    setDevice,
    trainJobId,
    trainProgress,
    startTraining,
    resetTraining,
    // Advanced training params
    qualityWeight, setQualityWeight,
    batchSize, setBatchSize,
    knnK, setKnnK,
    kmeansClusters, setKmeansClusters,
    svmKernel, setSvmKernel,
    svmGamma, setSvmGamma,
    mlpHiddenSizes, setMlpHiddenSizes,
    mlpEpochs, setMlpEpochs,
    mlpLearningRate, setMlpLearningRate,
    mlpDropout, setMlpDropout,

    // Config (Step 2)
    modelsPath,
    setModelsPath,
    decisions,
    addDecision,
    updateDecision,
    removeDecision,
    configJobId,
    configProgress,
    startConfigGeneration,

    // General
    actionLoading,
    actionError,
    clearError,
    reset,
  };
}
