import { useEffect } from 'react';
import { DIMENSION_INFO, STATUS_INFO, formatDuration } from '../../types/evaluation';
import { useProgress, useTask } from '../../hooks/useEvaluation';
import styles from './ProgressTracker.module.css';

interface ProgressTrackerProps {
  taskId: string;
  onComplete?: () => void;
  onCancel?: () => void;
}

export function ProgressTracker({ taskId, onComplete, onCancel }: ProgressTrackerProps) {
  const { task, refresh: refreshTask } = useTask(taskId);
  const { progress, connected, completed, error } = useProgress(taskId, task?.status === 'running');

  useEffect(() => {
    if (completed) {
      refreshTask();
      onComplete?.();
    }
  }, [completed, refreshTask, onComplete]);

  if (!task) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>
          <div className={styles.spinner} />
          <span>Loading task...</span>
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
            style={{ width: `${displayProgress.progress_percent}%` }}
          />
        </div>
        {displayProgress.message && (
          <p className={styles.progressMessage}>{displayProgress.message}</p>
        )}
      </div>

      <div className={styles.details}>
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
            {connected ? 'Connected' : 'Disconnected'}
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

      {task.error_message && (
        <div className={styles.taskError}>
          <h4>Error</h4>
          <p>{task.error_message}</p>
        </div>
      )}

      {task.status === 'running' && onCancel && (
        <div className={styles.actions}>
          <button className={styles.cancelButton} onClick={onCancel}>
            Cancel Evaluation
          </button>
        </div>
      )}
    </div>
  );
}

export default ProgressTracker;
