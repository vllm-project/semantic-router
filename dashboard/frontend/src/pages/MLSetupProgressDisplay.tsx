import { useRef } from 'react';
import { useMLPipelineWizard } from '../hooks/useMLPipeline';
import styles from './MLSetupPage.module.css';

interface MLSetupProgressDisplayProps {
  progress: ReturnType<typeof useMLPipelineWizard>['benchmarkProgress']['progress'];
  job: ReturnType<typeof useMLPipelineWizard>['benchmarkProgress']['job'];
  completed: boolean;
}

export default function MLSetupProgressDisplay({
  progress,
  job,
}: MLSetupProgressDisplayProps) {
  const rawPercent = progress?.percent ?? job?.progress ?? 0;
  const rawStep = progress?.step ?? job?.current_step ?? '';
  const rawMessage = progress?.message ?? '';
  const isFailed = job?.status === 'failed';
  const isComplete = job?.status === 'completed';

  // Never let displayed progress go backwards (prevents flicker during SSE reconnect)
  const highWaterRef = useRef({ percent: 0, step: '', message: '' });
  if (rawPercent > highWaterRef.current.percent || isComplete || isFailed) {
    highWaterRef.current = { percent: rawPercent, step: rawStep, message: rawMessage };
  }
  // Reset high water mark when job resets (new job at 0%)
  if (rawPercent === 0 && highWaterRef.current.percent >= 100) {
    highWaterRef.current = { percent: 0, step: '', message: '' };
  }

  const percent = highWaterRef.current.percent;
  const step = highWaterRef.current.step;
  const message = highWaterRef.current.message;

  return (
    <div className={styles.progressContainer}>
      <div className={styles.progressHeader}>
        <span className={styles.progressLabel}>{step || 'Processing...'}</span>
        <span className={styles.progressPercent}>{percent}%</span>
      </div>
      <div className={styles.progressBar}>
        <div
          className={`${styles.progressFill} ${
            isComplete ? styles.progressFillComplete : ''
          } ${isFailed ? styles.progressFillFailed : ''}`}
          style={{ width: `${percent}%` }}
        />
      </div>
      {message && <div className={styles.progressMessage}>{message}</div>}
    </div>
  );
}
