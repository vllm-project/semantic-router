import { useRef, useState } from 'react';
import { useMLPipelineWizard } from '../hooks/useMLPipeline';
import MLSetupProgressDisplay from './MLSetupProgressDisplay';
import styles from './MLSetupPage.module.css';

interface MLSetupBenchmarkStepProps {
  wizard: ReturnType<typeof useMLPipelineWizard>;
}

export default function MLSetupBenchmarkStep({ wizard }: MLSetupBenchmarkStepProps) {
  const modelsInputRef = useRef<HTMLInputElement>(null);
  const queriesInputRef = useRef<HTMLInputElement>(null);

  const jobDone =
    wizard.benchmarkProgress.completed && wizard.benchmarkProgress.job?.status === 'completed';
  const jobFailed = wizard.benchmarkProgress.job?.status === 'failed';

  return (
    <div className={styles.stepContent}>
      <h2 className={styles.stepTitle}>Step 1: Benchmark Your Models</h2>
      <p className={styles.stepDescription}>
        Upload your model configuration (YAML) and benchmark queries (JSONL) to collect embedding
        data from your models. This data will be used to train the ML classifiers.
      </p>

      {/* Skip benchmark option */}
      <div className={styles.skipBanner}>
        <div className={styles.skipBannerText}>
          <strong>Already have benchmark data?</strong> You can skip this step and upload your
          training data JSONL directly in Step 2.
        </div>
        <button className={styles.btnSecondary} onClick={() => wizard.goToStep(1)}>
          Skip to Training
        </button>
      </div>

      {/* Models YAML upload */}
      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Models Configuration (YAML)</label>
        <input
          ref={modelsInputRef}
          type="file"
          accept=".yaml,.yml"
          className={styles.fileInput}
          onChange={(e) => wizard.setModelsFile(e.target.files?.[0] || null)}
        />
        <div
          className={`${styles.fileDropZone} ${wizard.modelsFile ? styles.fileDropZoneActive : ''}`}
          onClick={() => modelsInputRef.current?.click()}
        >
          {wizard.modelsFile ? (
            <div className={styles.fileDropName}>{wizard.modelsFile.name}</div>
          ) : (
            <>
              <div className={styles.fileDropIcon}>
                <svg
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </div>
              <div className={styles.fileDropText}>Click to upload models YAML</div>
            </>
          )}
        </div>
      </div>

      {/* Queries JSONL upload */}
      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Benchmark Queries (JSONL)</label>
        <input
          ref={queriesInputRef}
          type="file"
          accept=".jsonl"
          className={styles.fileInput}
          onChange={(e) => wizard.setQueriesFile(e.target.files?.[0] || null)}
        />
        <div
          className={`${styles.fileDropZone} ${wizard.queriesFile ? styles.fileDropZoneActive : ''}`}
          onClick={() => queriesInputRef.current?.click()}
        >
          {wizard.queriesFile ? (
            <div className={styles.fileDropName}>{wizard.queriesFile.name}</div>
          ) : (
            <>
              <div className={styles.fileDropIcon}>
                <svg
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </div>
              <div className={styles.fileDropText}>Click to upload queries JSONL</div>
            </>
          )}
        </div>
      </div>

      {/* Concurrency */}
      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Concurrency</label>
        <input
          type="number"
          className={styles.numberInput}
          min={1}
          max={32}
          value={wizard.benchmarkConfig.concurrency}
          onChange={(e) =>
            wizard.setBenchmarkConfig({
              ...wizard.benchmarkConfig,
              concurrency: parseInt(e.target.value) || 4,
            })
          }
        />
        <div className={styles.formHint}>Number of concurrent requests during benchmark</div>
      </div>

      {/* Advanced Benchmark Settings */}
      <MLSetupBenchmarkAdvancedSettings wizard={wizard} />

      {/* Run button or progress */}
      {!wizard.benchmarkJobId && (
        <button
          className={styles.btnPrimary}
          onClick={wizard.startBenchmark}
          disabled={wizard.actionLoading || !wizard.modelsFile || !wizard.queriesFile}
        >
          {wizard.actionLoading ? 'Starting...' : 'Run Benchmark'}
        </button>
      )}

      {/* Progress */}
      {wizard.benchmarkJobId && (
        <MLSetupProgressDisplay
          progress={wizard.benchmarkProgress.progress}
          job={wizard.benchmarkProgress.job}
          completed={wizard.benchmarkProgress.completed}
        />
      )}

      {/* Success */}
      {jobDone && (
        <div className={styles.successCard}>
          <div className={styles.successIcon}>
            <svg
              width="48"
              height="48"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#22c55e"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <polyline points="16 8 10 16 7 13" />
            </svg>
          </div>
          <div className={styles.successTitle}>Benchmark Complete</div>
          <div className={styles.successMessage}>
            Embedding data has been collected. Proceed to the next step to train ML classifiers.
          </div>
        </div>
      )}

      {jobFailed && (
        <div className={styles.errorAlert}>
          <span>
            Benchmark failed: {wizard.benchmarkProgress.job?.error || 'Unknown error'}
          </span>
        </div>
      )}
    </div>
  );
}

function MLSetupBenchmarkAdvancedSettings({ wizard }: MLSetupBenchmarkStepProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className={styles.advancedSection}>
      <button className={styles.advancedToggle} onClick={() => setOpen(!open)}>
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          style={{
            transform: open ? 'rotate(90deg)' : 'rotate(0)',
            transition: 'transform 0.2s',
          }}
        >
          <polyline points="9 18 15 12 9 6" />
        </svg>
        Advanced Settings
      </button>
      {open && (
        <div className={styles.advancedContent}>
          <div className={styles.advancedGrid}>
            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Max Tokens</label>
              <input
                type="number"
                className={styles.numberInput}
                min={1}
                max={8192}
                value={wizard.benchmarkConfig.max_tokens ?? 1024}
                onChange={(e) =>
                  wizard.setBenchmarkConfig({
                    ...wizard.benchmarkConfig,
                    max_tokens: parseInt(e.target.value) || 1024,
                  })
                }
              />
              <div className={styles.formHint}>Maximum tokens in LLM response</div>
            </div>
            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Temperature</label>
              <input
                type="number"
                className={styles.numberInput}
                min={0}
                max={2}
                step={0.1}
                value={wizard.benchmarkConfig.temperature ?? 0}
                onChange={(e) =>
                  wizard.setBenchmarkConfig({
                    ...wizard.benchmarkConfig,
                    temperature: parseFloat(e.target.value) || 0,
                  })
                }
              />
              <div className={styles.formHint}>Generation temperature (0 = deterministic)</div>
            </div>
            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Query Limit</label>
              <input
                type="number"
                className={styles.numberInput}
                min={0}
                value={wizard.benchmarkConfig.limit ?? 0}
                onChange={(e) =>
                  wizard.setBenchmarkConfig({
                    ...wizard.benchmarkConfig,
                    limit: parseInt(e.target.value) || 0,
                  })
                }
              />
              <div className={styles.formHint}>Limit queries to process (0 = all)</div>
            </div>
            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Concise Prompts</label>
              <div className={styles.deviceSelector}>
                <button
                  className={`${styles.deviceOption} ${wizard.benchmarkConfig.concise ? styles.deviceOptionSelected : ''}`}
                  onClick={() =>
                    wizard.setBenchmarkConfig({ ...wizard.benchmarkConfig, concise: true })
                  }
                >
                  On
                </button>
                <button
                  className={`${styles.deviceOption} ${!wizard.benchmarkConfig.concise ? styles.deviceOptionSelected : ''}`}
                  onClick={() =>
                    wizard.setBenchmarkConfig({ ...wizard.benchmarkConfig, concise: false })
                  }
                >
                  Off
                </button>
              </div>
              <div className={styles.formHint}>Shorter prompts for faster inference</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
