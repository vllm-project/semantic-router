// Custom hooks for evaluation functionality

import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  EvaluationTask,
  DatasetInfo,
  ProgressUpdate,
  TaskResults,
  CreateTaskRequest,
  EvaluationDimension,
} from '../types/evaluation';
import * as api from '../utils/evaluationApi';
import { useReadonly } from '../contexts/ReadonlyContext';

// Hook for managing tasks list
export function useTasks(autoRefresh = false, refreshInterval = 5000) {
  const [tasks, setTasks] = useState<EvaluationTask[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchTasks = useCallback(async () => {
    try {
      const data = await api.listTasks();
      setTasks(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch tasks');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchTasks();

    if (autoRefresh) {
      const interval = setInterval(fetchTasks, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [fetchTasks, autoRefresh, refreshInterval]);

  const refresh = useCallback(() => {
    setLoading(true);
    fetchTasks();
  }, [fetchTasks]);

  return { tasks, loading, error, refresh };
}

// Hook for a single task with progress tracking
export function useTask(taskId: string | null) {
  const [task, setTask] = useState<EvaluationTask | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchTask = useCallback(async () => {
    if (!taskId) return;
    setLoading(true);
    try {
      const data = await api.getTask(taskId);
      setTask(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch task');
    } finally {
      setLoading(false);
    }
  }, [taskId]);

  useEffect(() => {
    fetchTask();
  }, [fetchTask]);

  return { task, loading, error, refresh: fetchTask };
}

// Hook for progress tracking via SSE
export function useProgress(taskId: string | null, enabled = true) {
  const [progress, setProgress] = useState<ProgressUpdate | null>(null);
  const [connected, setConnected] = useState(false);
  const [completed, setCompleted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!taskId || !enabled) {
      return;
    }

    const cleanup = api.subscribeToProgress(
      taskId,
      (update) => {
        setProgress(update);
        setConnected(true);
        setError(null);
      },
      () => {
        setCompleted(true);
        setConnected(false);
      },
      (err) => {
        setError(err.message);
        setConnected(false);
      }
    );

    cleanupRef.current = cleanup;

    return () => {
      cleanup();
      cleanupRef.current = null;
    };
  }, [taskId, enabled]);

  const disconnect = useCallback(() => {
    if (cleanupRef.current) {
      cleanupRef.current();
      cleanupRef.current = null;
      setConnected(false);
    }
  }, []);

  return { progress, connected, completed, error, disconnect };
}

// Hook for task results
export function useResults(taskId: string | null) {
  const [results, setResults] = useState<TaskResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchResults = useCallback(async () => {
    if (!taskId) return;
    setLoading(true);
    try {
      const data = await api.getResults(taskId);
      setResults(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch results');
    } finally {
      setLoading(false);
    }
  }, [taskId]);

  useEffect(() => {
    fetchResults();
  }, [fetchResults]);

  return { results, loading, error, refresh: fetchResults };
}

// Hook for available datasets
export function useDatasets() {
  const [datasets, setDatasets] = useState<Record<string, DatasetInfo[]>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchDatasets() {
      try {
        const data = await api.getDatasets();
        setDatasets(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch datasets');
      } finally {
        setLoading(false);
      }
    }
    fetchDatasets();
  }, []);

  return { datasets, loading, error };
}

// Hook for task mutations (create, run, cancel, delete)
export function useTaskMutations() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const createTask = useCallback(async (request: CreateTaskRequest): Promise<EvaluationTask | null> => {
    setLoading(true);
    setError(null);
    try {
      const task = await api.createTask(request);
      return task;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create task');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const runTask = useCallback(async (taskId: string): Promise<boolean> => {
    setLoading(true);
    setError(null);
    try {
      await api.runTask({ task_id: taskId });
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to run task');
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const cancelTask = useCallback(async (taskId: string): Promise<boolean> => {
    setLoading(true);
    setError(null);
    try {
      await api.cancelTask(taskId);
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel task');
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const deleteTask = useCallback(async (taskId: string): Promise<boolean> => {
    setLoading(true);
    setError(null);
    try {
      await api.deleteTask(taskId);
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete task');
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const clearError = useCallback(() => setError(null), []);

  return {
    loading,
    error,
    createTask,
    runTask,
    cancelTask,
    deleteTask,
    clearError,
  };
}

// Hook for managing task creation form state
export function useTaskCreationForm() {
  const { envoyUrl } = useReadonly();
  const [step, setStep] = useState(1);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [dimensions, setDimensions] = useState<EvaluationDimension[]>(['hallucination']);
  const [selectedDatasets, setSelectedDatasets] = useState<Record<string, string[]>>({});
  const [maxSamples, setMaxSamples] = useState(50);
  const [endpoint, setEndpoint] = useState(envoyUrl || 'http://localhost:8801');
  const [model, setModel] = useState('MoM');
  const [concurrent, setConcurrent] = useState(1);
  const [samplesPerCat, setSamplesPerCat] = useState(10);

  // Update endpoint when envoyUrl becomes available
  useEffect(() => {
    if (envoyUrl && endpoint === 'http://localhost:8801') {
      setEndpoint(envoyUrl);
    }
  }, [envoyUrl, endpoint]);

  const toggleDimension = useCallback((dim: EvaluationDimension) => {
    setDimensions((prev) => {
      if (prev.includes(dim)) {
        return prev.filter((d) => d !== dim);
      }
      return [...prev, dim];
    });
  }, []);

  const toggleDataset = useCallback((dimension: string, dataset: string) => {
    setSelectedDatasets((prev) => {
      const current = prev[dimension] || [];
      if (current.includes(dataset)) {
        return { ...prev, [dimension]: current.filter((d) => d !== dataset) };
      }
      return { ...prev, [dimension]: [...current, dataset] };
    });
  }, []);

  const nextStep = useCallback(() => setStep((s) => Math.min(s + 1, 4)), []);
  const prevStep = useCallback(() => setStep((s) => Math.max(s - 1, 1)), []);
  const goToStep = useCallback((s: number) => setStep(s), []);

  const getConfig = useCallback((): CreateTaskRequest => {
    // Ensure all selected dimensions have at least default datasets
    const datasets: Record<string, string[]> = {};
    for (const dim of dimensions) {
      datasets[dim] = selectedDatasets[dim]?.length > 0 ? selectedDatasets[dim] : ['default'];
    }

    return {
      name,
      description,
      config: {
        dimensions,
        datasets,
        max_samples: maxSamples,
        endpoint,
        model,
        concurrent,
        samples_per_cat: samplesPerCat,
      },
    };
  }, [name, description, dimensions, selectedDatasets, maxSamples, endpoint, model, concurrent, samplesPerCat]);

  const reset = useCallback(() => {
    setStep(1);
    setName('');
    setDescription('');
    setDimensions(['hallucination']);
    setSelectedDatasets({});
    setMaxSamples(50);
    setEndpoint(envoyUrl || 'http://localhost:8801');
    setModel('MoM');
    setConcurrent(1);
    setSamplesPerCat(10);
  }, [envoyUrl]);

  const isStepValid = useCallback(
    (stepNum: number): boolean => {
      switch (stepNum) {
        case 1:
          return name.trim().length > 0;
        case 2:
          return dimensions.length > 0;
        case 3:
          return true; // Datasets are optional with defaults
        case 4:
          return true; // Review step
        default:
          return false;
      }
    },
    [name, dimensions]
  );

  return {
    step,
    name,
    setName,
    description,
    setDescription,
    dimensions,
    toggleDimension,
    selectedDatasets,
    toggleDataset,
    maxSamples,
    setMaxSamples,
    endpoint,
    setEndpoint,
    model,
    setModel,
    concurrent,
    setConcurrent,
    samplesPerCat,
    setSamplesPerCat,
    nextStep,
    prevStep,
    goToStep,
    getConfig,
    reset,
    isStepValid,
  };
}
