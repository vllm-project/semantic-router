// Custom hooks for evaluation functionality

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import type {
  EvaluationTask,
  DatasetInfo,
  ProgressUpdate,
  TaskResults,
  CreateTaskRequest,
  EvaluationDimension,
  EvaluationLevel,
} from '../types/evaluation';
import * as api from '../utils/evaluationApi';
import { useReadonly } from '../contexts/ReadonlyContext';
import {
  DEFAULT_ROUTER_EVAL_ENDPOINT,
  filterSelectedDatasetsByDimensions,
  getDefaultDimensionsForLevel,
  getDefaultEndpointForLevel,
  normalizeDimensionsForLevel,
} from '../utils/evaluationConfig';
import { CANONICAL_AUTO_MODEL } from '../utils/routerModelSelection';
import { createEvaluationRequest } from './evaluationRequestSupport';

interface EvaluationLoadOptions {
  showLoading?: boolean;
  allowHidden?: boolean;
}

interface VisibilityPollingOptions {
  active: boolean;
  autoRefresh: boolean;
  refreshInterval: number;
  load: (options?: EvaluationLoadOptions) => Promise<void>;
  invalidate: () => void;
  shouldPoll?: () => boolean;
}

const alwaysPoll = () => true;
const noop = () => undefined;

function useVisibilityPolling({
  active,
  autoRefresh,
  refreshInterval,
  load,
  invalidate,
  shouldPoll = alwaysPoll,
}: VisibilityPollingOptions) {
  useEffect(() => {
    if (!active) {
      invalidate();
      return;
    }

    void load({ showLoading: true, allowHidden: true });
    if (!autoRefresh) return invalidate;

    const refreshWhenVisible = () => {
      if (!document.hidden && shouldPoll()) void load();
    };
    const interval = window.setInterval(() => {
      if (shouldPoll()) void load();
    }, refreshInterval);
    document.addEventListener('visibilitychange', refreshWhenVisible);

    return () => {
      window.clearInterval(interval);
      document.removeEventListener('visibilitychange', refreshWhenVisible);
      invalidate();
    };
  }, [active, autoRefresh, invalidate, load, refreshInterval, shouldPoll]);
}

// Hook for managing tasks list
export function useTasks(autoRefresh = false, refreshInterval = 5000) {
  const [tasks, setTasks] = useState<EvaluationTask[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const request = useMemo(
    () => createEvaluationRequest((signal) => api.listTasks(undefined, signal)),
    [],
  );

  const fetchTasks = useCallback(async (options: EvaluationLoadOptions = {}) => {
    if (options.showLoading) setLoading(true);
    let settled = false;
    try {
      const data = await request.run({ allowHidden: options.allowHidden });
      if (!data) return;
      settled = true;
      setTasks(data);
      setError(null);
    } catch (err) {
      settled = true;
      setError(err instanceof Error ? err.message : 'Failed to fetch tasks');
    } finally {
      if (settled && options.showLoading) setLoading(false);
    }
  }, [request]);

  useVisibilityPolling({
    active: true,
    autoRefresh,
    refreshInterval,
    load: fetchTasks,
    invalidate: request.invalidate,
  });

  const refresh = useCallback(() => {
    return fetchTasks({ showLoading: true, allowHidden: true });
  }, [fetchTasks]);

  return { tasks, loading, error, refresh };
}

// Hook for a single task with progress tracking
export function useTask(taskId: string | null, autoRefresh = false, refreshInterval = 1000) {
  const [task, setTask] = useState<EvaluationTask | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const terminalRef = useRef(false);
  const request = useMemo(
    () =>
      taskId
        ? createEvaluationRequest((signal) => api.getTask(taskId, signal))
        : null,
    [taskId],
  );

  const fetchTask = useCallback(async (options: EvaluationLoadOptions = {}) => {
    if (!request) return;
    if (options.showLoading) setLoading(true);
    let settled = false;
    try {
      const data = await request.run({ allowHidden: options.allowHidden });
      if (!data) return;
      settled = true;
      terminalRef.current = ['completed', 'failed', 'cancelled'].includes(data.status);
      setTask(data);
      setError(null);
    } catch (err) {
      settled = true;
      setError(err instanceof Error ? err.message : 'Failed to fetch task');
    } finally {
      if (settled && options.showLoading) setLoading(false);
    }
  }, [request]);

  useEffect(() => {
    terminalRef.current = false;
    setTask(null);
    setError(null);
    setLoading(Boolean(taskId));
  }, [taskId]);

  const shouldPoll = useCallback(() => !terminalRef.current, []);
  useVisibilityPolling({
    active: Boolean(taskId && request),
    autoRefresh,
    refreshInterval,
    load: fetchTask,
    invalidate: request?.invalidate ?? noop,
    shouldPoll,
  });

  const refresh = useCallback(
    () => fetchTask({ showLoading: true, allowHidden: true }),
    [fetchTask],
  );
  return { task: task?.id === taskId ? task : null, loading, error, refresh };
}

// Hook for progress tracking via SSE
export function useProgress(taskId: string | null, enabled = true) {
  const [progress, setProgress] = useState<ProgressUpdate | null>(null);
  const [connected, setConnected] = useState(false);
  const [completed, setCompleted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);
  const disconnectRequestedRef = useRef(false);

  useEffect(() => {
    disconnectRequestedRef.current = false;
    setProgress(null);
    setCompleted(false);
    setConnected(false);
    setError(null);
    if (!taskId || !enabled) {
      return;
    }

    let streamCleanup: (() => void) | null = null;
    let terminal = false;

    const closeStream = (updateState = true) => {
      streamCleanup?.();
      streamCleanup = null;
      cleanupRef.current = null;
      if (updateState) setConnected(false);
    };
    const connect = () => {
      if (document.hidden || terminal || disconnectRequestedRef.current || streamCleanup) return;
      streamCleanup = api.subscribeToProgress(
        taskId,
        (update) => {
          setProgress(update);
          setConnected(true);
          setError(null);
        },
        () => {
          terminal = true;
          disconnectRequestedRef.current = true;
          streamCleanup = null;
          cleanupRef.current = null;
          setCompleted(true);
          setConnected(false);
        },
        (err) => {
          streamCleanup = null;
          cleanupRef.current = null;
          setError(err.message);
          setConnected(false);
        },
        () => {
          setConnected(true);
          setError(null);
        },
      );
      cleanupRef.current = streamCleanup;
    };
    const handleVisibilityChange = () => {
      if (document.hidden) closeStream();
      else connect();
    };

    connect();
    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      terminal = true;
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      closeStream(false);
    };
  }, [taskId, enabled]);

  const disconnect = useCallback(() => {
    disconnectRequestedRef.current = true;
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

  const request = useMemo(
    () =>
      taskId
        ? createEvaluationRequest((signal) => api.getResults(taskId, signal))
        : null,
    [taskId],
  );

  const fetchResults = useCallback(async (options: EvaluationLoadOptions = {}) => {
    if (!request) return;
    if (options.showLoading) setLoading(true);
    let settled = false;
    try {
      const data = await request.run({ allowHidden: options.allowHidden });
      if (!data) return;
      settled = true;
      setResults(data);
      setError(null);
    } catch (err) {
      settled = true;
      setError(err instanceof Error ? err.message : 'Failed to fetch results');
    } finally {
      if (settled && options.showLoading) setLoading(false);
    }
  }, [request]);

  useEffect(() => {
    setResults(null);
    setError(null);
    setLoading(Boolean(taskId));
    if (!request) return;
    void fetchResults({ showLoading: true, allowHidden: true });
    return request.invalidate;
  }, [fetchResults, request, taskId]);

  const refresh = useCallback(
    () => fetchResults({ showLoading: true, allowHidden: true }),
    [fetchResults],
  );
  return {
    results: results?.task.id === taskId ? results : null,
    loading,
    error,
    refresh,
  };
}

// Hook for available datasets
export function useDatasets() {
  const [datasets, setDatasets] = useState<Record<string, DatasetInfo[]>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const request = useMemo(
    () => createEvaluationRequest((signal) => api.getDatasets(signal)),
    [],
  );
  const fetchDatasets = useCallback(async (options: EvaluationLoadOptions = {}) => {
    if (options.showLoading) setLoading(true);
    let settled = false;
    try {
      const data = await request.run({ allowHidden: options.allowHidden });
      if (!data) return;
      settled = true;
      setDatasets(data);
      setError(null);
    } catch (err) {
      settled = true;
      setError(err instanceof Error ? err.message : 'Failed to fetch datasets');
    } finally {
      if (settled && options.showLoading) setLoading(false);
    }
  }, [request]);

  useEffect(() => {
    void fetchDatasets({ showLoading: true, allowHidden: true });
    return request.invalidate;
  }, [fetchDatasets, request]);

  const refresh = useCallback(
    () => fetchDatasets({ showLoading: true, allowHidden: true }),
    [fetchDatasets],
  );
  return { datasets, loading, error, refresh };
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
  const { envoyUrl, routerEvalEndpoint } = useReadonly();
  const [step, setStep] = useState(1);
  const [level, setLevelState] = useState<EvaluationLevel>('router');
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [dimensions, setDimensions] = useState<EvaluationDimension[]>(getDefaultDimensionsForLevel('router'));
  const [selectedDatasets, setSelectedDatasets] = useState<Record<string, string[]>>({});
  const [maxSamples, setMaxSamples] = useState(50);
  const [endpoint, setEndpoint] = useState(DEFAULT_ROUTER_EVAL_ENDPOINT);
  const [model, setModel] = useState(CANONICAL_AUTO_MODEL);
  const [concurrent, setConcurrent] = useState(1);
  const [samplesPerCat, setSamplesPerCat] = useState(10);

  // Update endpoint based on level
  useEffect(() => {
    setEndpoint(getDefaultEndpointForLevel(level, routerEvalEndpoint, envoyUrl));
  }, [level, routerEvalEndpoint, envoyUrl]);

  const setLevel = useCallback((nextLevel: EvaluationLevel) => {
    setLevelState(nextLevel);
    setDimensions((prevDimensions) => {
      const nextDimensions = normalizeDimensionsForLevel(nextLevel, prevDimensions);
      setSelectedDatasets((prevSelectedDatasets) =>
        filterSelectedDatasetsByDimensions(prevSelectedDatasets, nextDimensions)
      );
      return nextDimensions;
    });
  }, []);

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
    const normalizedDimensions = normalizeDimensionsForLevel(level, dimensions);

    const datasets: Record<string, string[]> = {};
    const filteredSelectedDatasets = filterSelectedDatasetsByDimensions(selectedDatasets, normalizedDimensions);
    for (const dim of normalizedDimensions) {
      datasets[dim] = filteredSelectedDatasets[dim] ?? [];
    }

    return {
      name,
      description,
      config: {
        level,
        dimensions: normalizedDimensions,
        datasets,
        max_samples: maxSamples,
        endpoint,
        model,
        concurrent,
        samples_per_cat: samplesPerCat,
      },
    };
  }, [level, name, description, dimensions, selectedDatasets, maxSamples, endpoint, model, concurrent, samplesPerCat]);

  const reset = useCallback(() => {
    setStep(1);
    setLevelState('router');
    setName('');
    setDescription('');
    setDimensions(getDefaultDimensionsForLevel('router'));
    setSelectedDatasets({});
    setMaxSamples(50);
    setEndpoint(getDefaultEndpointForLevel('router', routerEvalEndpoint, envoyUrl));
    setModel(CANONICAL_AUTO_MODEL);
    setConcurrent(1);
    setSamplesPerCat(10);
  }, [envoyUrl, routerEvalEndpoint]);

  const isStepValid = useCallback(
    (stepNum: number): boolean => {
      switch (stepNum) {
        case 1:
          return level.trim().length > 0 && name.trim().length > 0;
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
    [level, name, dimensions]
  );

  return {
    step,
    level,
    setLevel,
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
