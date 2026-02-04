import { useCallback } from 'react';
import type { EvaluationTask } from '../../types/evaluation';
import { STATUS_INFO, formatDate, formatDuration } from '../../types/evaluation';
import styles from './TaskList.module.css';

interface TaskListProps {
  tasks: EvaluationTask[];
  loading: boolean;
  onView: (task: EvaluationTask) => void;
  onRun: (task: EvaluationTask) => void;
  onCancel: (task: EvaluationTask) => void;
  onDelete: (task: EvaluationTask) => void;
  onRefresh: () => void;
}

export function TaskList({ tasks, loading, onView, onRun, onCancel, onDelete, onRefresh }: TaskListProps) {

  const getStatusBadge = useCallback((status: EvaluationTask['status']) => {
    const info = STATUS_INFO[status];
    return (
      <span
        className={styles.statusBadge}
        style={{ color: info.color, backgroundColor: info.bgColor }}
      >
        {info.label}
      </span>
    );
  }, []);

  const canRun = useCallback((task: EvaluationTask) => {
    return task.status === 'pending' || task.status === 'failed';
  }, []);

  const canCancel = useCallback((task: EvaluationTask) => {
    return task.status === 'running';
  }, []);

  if (loading && tasks.length === 0) {
    return (
      <div className={styles.loading}>
        <div className={styles.spinner} />
        <span>Loading tasks...</span>
      </div>
    );
  }

  if (tasks.length === 0) {
    return (
      <div className={styles.empty}>
        <div className={styles.emptyIcon}>📋</div>
        <h3>No Evaluation Tasks</h3>
        <p>Create a new evaluation task to get started.</p>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h3>Evaluation Tasks</h3>
        <button className={styles.refreshButton} onClick={onRefresh} disabled={loading}>
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      <div className={styles.tableWrapper}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Name</th>
              <th>Status</th>
              <th>Progress</th>
              <th>Created</th>
              <th>Duration</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {tasks.map((task) => (
              <tr key={task.id}>
                <td>
                  <div className={styles.taskName}>
                    <span className={styles.name}>{task.name}</span>
                    {task.description && (
                      <span className={styles.description}>{task.description}</span>
                    )}
                  </div>
                </td>
                <td>{getStatusBadge(task.status)}</td>
                <td>
                  <div className={styles.progress}>
                    <div
                      className={styles.progressBar}
                      style={{ width: `${task.progress_percent}%` }}
                    />
                    <span className={styles.progressText}>{task.progress_percent}%</span>
                  </div>
                  {task.current_step && (
                    <span className={styles.currentStep}>{task.current_step}</span>
                  )}
                </td>
                <td className={styles.date}>{formatDate(task.created_at)}</td>
                <td className={styles.duration}>
                  {formatDuration(task.started_at, task.completed_at)}
                </td>
                <td>
                  <div className={styles.actions}>
                    <button
                      className={styles.actionButton}
                      onClick={() => onView(task)}
                      title="View Details"
                    >
                      View
                    </button>
                    {canRun(task) && (
                      <button
                        className={`${styles.actionButton} ${styles.runButton}`}
                        onClick={() => onRun(task)}
                        title="Run Evaluation"
                      >
                        Run
                      </button>
                    )}
                    {canCancel(task) && (
                      <button
                        className={`${styles.actionButton} ${styles.cancelButton}`}
                        onClick={() => onCancel(task)}
                        title="Cancel Evaluation"
                      >
                        Cancel
                      </button>
                    )}
                    <button
                      className={`${styles.actionButton} ${styles.deleteButton}`}
                      onClick={() => onDelete(task)}
                      title="Delete Task"
                      disabled={task.status === 'running'}
                    >
                      Delete
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default TaskList;
