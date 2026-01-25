import { useState, useMemo } from 'react';
import type { EvaluationTask } from '../../types/evaluation';
import { STATUS_INFO, formatDate, formatDuration } from '../../types/evaluation';
import styles from './HistoricalResults.module.css';

interface HistoricalResultsProps {
  tasks: EvaluationTask[];
  onViewResults: (task: EvaluationTask) => void;
  onCompare?: (tasks: EvaluationTask[]) => void;
}

export function HistoricalResults({ tasks, onViewResults, onCompare }: HistoricalResultsProps) {
  const [selectedTasks, setSelectedTasks] = useState<Set<string>>(new Set());
  const [sortBy, setSortBy] = useState<'date' | 'name'>('date');
  const [filterStatus, setFilterStatus] = useState<string>('all');

  const completedTasks = useMemo(() => {
    return tasks.filter(t => t.status === 'completed' || t.status === 'failed');
  }, [tasks]);

  const filteredTasks = useMemo(() => {
    let filtered = completedTasks;

    if (filterStatus !== 'all') {
      filtered = filtered.filter(t => t.status === filterStatus);
    }

    return filtered.sort((a, b) => {
      if (sortBy === 'date') {
        return new Date(b.completed_at || b.created_at).getTime() -
               new Date(a.completed_at || a.created_at).getTime();
      }
      return a.name.localeCompare(b.name);
    });
  }, [completedTasks, filterStatus, sortBy]);

  const toggleTask = (taskId: string) => {
    setSelectedTasks(prev => {
      const next = new Set(prev);
      if (next.has(taskId)) {
        next.delete(taskId);
      } else {
        next.add(taskId);
      }
      return next;
    });
  };

  const handleCompare = () => {
    const tasksToCompare = filteredTasks.filter(t => selectedTasks.has(t.id));
    onCompare?.(tasksToCompare);
  };

  if (completedTasks.length === 0) {
    return (
      <div className={styles.empty}>
        <div className={styles.emptyIcon}>📊</div>
        <h3>No Historical Results</h3>
        <p>Complete an evaluation to see results here.</p>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.toolbar}>
        <div className={styles.filters}>
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className={styles.select}
          >
            <option value="all">All Status</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as 'date' | 'name')}
            className={styles.select}
          >
            <option value="date">Sort by Date</option>
            <option value="name">Sort by Name</option>
          </select>
        </div>
        {onCompare && selectedTasks.size >= 2 && (
          <button className={styles.compareButton} onClick={handleCompare}>
            Compare Selected ({selectedTasks.size})
          </button>
        )}
      </div>

      <div className={styles.list}>
        {filteredTasks.map((task) => {
          const statusInfo = STATUS_INFO[task.status];
          const isSelected = selectedTasks.has(task.id);

          return (
            <div
              key={task.id}
              className={`${styles.card} ${isSelected ? styles.selected : ''}`}
            >
              {onCompare && (
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => toggleTask(task.id)}
                  className={styles.checkbox}
                />
              )}
              <div className={styles.cardContent}>
                <div className={styles.cardHeader}>
                  <h4>{task.name}</h4>
                  <span
                    className={styles.statusBadge}
                    style={{ color: statusInfo.color, backgroundColor: statusInfo.bgColor }}
                  >
                    {statusInfo.label}
                  </span>
                </div>
                {task.description && (
                  <p className={styles.description}>{task.description}</p>
                )}
                <div className={styles.cardMeta}>
                  <span>Completed: {formatDate(task.completed_at)}</span>
                  <span>Duration: {formatDuration(task.started_at, task.completed_at)}</span>
                  <span>Dimensions: {task.config.dimensions.length}</span>
                </div>
              </div>
              <button
                className={styles.viewButton}
                onClick={() => onViewResults(task)}
              >
                View Results
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default HistoricalResults;
