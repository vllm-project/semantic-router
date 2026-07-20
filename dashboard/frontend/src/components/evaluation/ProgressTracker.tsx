import { useEffect } from 'react';
import { DIMENSION_INFO, STATUS_INFO, LEVEL_INFO, formatDuration } from '../../types/evaluation';
import { useProgress, useTask } from '../../hooks/useEvaluation';
import styles from './ProgressTracker.module.css';

interface ProgressTrackerProps {
  taskId: string;
  onComplete?: () => void;
  onCancel?: () => void;
}

export function ProgressTracker({ taskId, onComplete, onCancel }: ProgressTrackerProps) {
  const { task, loading, error: taskError, refresh: refreshTask } = useTask(taskId, true, 1000);
  const { progress, connected, completed, error, disconnect } = useProgress(
    taskId,
    task?.status === 'running',
  );

  useEffect(() => {
    if (completed) {
      refreshTask();
    }
  }, [completed, refreshTask]);

  useEffect(() => {
    if (!task) {
      return;
    }

    if (task.status === 'completed') {
      disconnect();
      onComplete?.();
      return;
    }

    if (task.status === 'failed' || task.status === 'cancelled') {
      disconnect();
    }
  }, [task, disconnect, onComplete]);

  if (taskError && !task) {
    return (
      <div className={styles.container}>
        <div className={styles.loadError} role="alert">
          <h3>Evaluation progress is unavailable</h3>
          <p>{taskError}</p>
          <button type="button" onClick={() => void refreshTask()}>Retry</button>
        </div>
      </div>
    );
  }

  if (!task) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>
          <div className={styles.spinner} />
          <span>{loading ? 'Loading task…' : 'Waiting for task state…'}</span>
        </div>
      </div>
    );
  }

  const displayProgress = progress || {
    progress_percent: task.progress_percent,
    current_step: task.current_step || '',
    message: '',
  };

  const statusInfo = STATUS_INFO[task.status];
  const streamLabel =
    task.status === 'running' ? (connected ? 'Connected' : 'Reconnecting') : 'Not active';

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.taskInfo}>
          <h3>{task.name}</h3>
          {task.description && <p className={styles.description}>{task.description}</p>}
        </div>
        <span
          className={styles.statusBadge}
          style={{ color: statusInfo.color, backgroundColor: statusInfo.bgColor }}
        >
          {statusInfo.label}
        </span>
      </div>

      <div className={styles.progressSection}>
        <div className={styles.progressHeader}>
          <span className={styles.progressLabel}>
            {displayProgress.current_step || 'Preparing...'}
          </span>
          <span className={styles.progressPercent}>{displayProgress.progress_percent}%</span>
        </div>
        <div className={styles.progressBar}>
          <div
            className={`${styles.progressFill} ${task.status === 'running' ? styles.animated : ''}`}
            style={{ width: `${Math.min(100, Math.max(0, displayProgress.progress_percent))}%` }}
          />
        </div>
        {displayProgress.message && (
          <p className={styles.progressMessage}>{displayProgress.message}</p>
        )}
      </div>

      <div className={styles.details}>
        <div className={styles.detailItem}>
          <span className={styles.detailLabel}>Level</span>
          <span className={styles.detailValue} style={{ color: LEVEL_INFO[task.config.level].color }}>
            {LEVEL_INFO[task.config.level].label}
          </span>
        </div>
        <div className={styles.detailItem}>
          <span className={styles.detailLabel}>Started</span>
          <span className={styles.detailValue}>
            {task.started_at ? new Date(task.started_at).toLocaleTimeString() : '-'}
          </span>
        </div>
        <div className={styles.detailItem}>
          <span className={styles.detailLabel}>Duration</span>
          <span className={styles.detailValue}>
            {formatDuration(task.started_at, task.completed_at)}
          </span>
        </div>
        <div className={styles.detailItem}>
          <span className={styles.detailLabel}>Connection</span>
          <span className={`${styles.detailValue} ${connected ? styles.connected : styles.disconnected}`}>
            {streamLabel}
          </span>
        </div>
      </div>

      <div className={styles.dimensions}>
        <h4>Evaluation Dimensions</h4>
        <div className={styles.dimensionList}>
          {task.config.dimensions.map((dim) => {
            const info = DIMENSION_INFO[dim];
            const isActive = displayProgress.current_step?.toLowerCase().includes(dim);
            return (
              <div
                key={dim}
                className={`${styles.dimension} ${isActive ? styles.activeDimension : ''}`}
                style={{ '--dim-color': info.color } as React.CSSProperties}
              >
                <span className={styles.dimensionIndicator} style={{ backgroundColor: info.color }} />
                <span className={styles.dimensionName}>{info.label}</span>
              </div>
            );
          })}
        </div>
      </div>

      {error && (
        <div className={styles.error}>
          <span>Connection error: {error}</span>
        </div>
      )}

      {taskError && (
        <div className={styles.error} role="alert">
          <span>Task refresh error: {taskError}</span>
        </div>
      )}

      {task.error_message && (
        <div className={styles.taskError}>
          <h4>Error</h4>
          <p>{task.error_message}</p>
        </div>
      )}

      {task.status === 'running' && onCancel && (
        <div className={styles.actions}>
          <button type="button" className={styles.cancelButton} onClick={onCancel}>
            Cancel Evaluation
          </button>
        </div>
      )}
    </div>
  );
}

export default ProgressTracker;
