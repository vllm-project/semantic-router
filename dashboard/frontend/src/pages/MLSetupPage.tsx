import React, { useEffect, useRef, useState } from 'react';
import styles from './MLSetupPage.module.css';
import { useMLPipelineWizard } from '../hooks/useMLPipeline';
import {
  ML_ALGORITHM_INFO,
  PIPELINE_STEPS,
  type MLAlgorithm,
  type SvmKernel,
} from '../types/mlPipeline';
import { getDownloadUrl } from '../utils/mlPipelineApi';

const MLSetupPage: React.FC = () => {
  const wizard = useMLPipelineWizard();

  return (
    <div className={styles.page}>
      {/* Header */}
      <div className={styles.header}>
        <h1 className={styles.title}>ML Model Selection Setup</h1>
        <p className={styles.subtitle}>
          Configure intelligent model routing by benchmarking your models, training ML classifiers,
          and generating deployment-ready configuration.
        </p>
      </div>

      {/* Stepper */}
      <div className={styles.stepper}>
        {PIPELINE_STEPS.map((step, idx) => (
          <React.Fragment key={step.key}>
            {idx > 0 && (
              <div
                className={`${styles.stepConnector} ${
                  idx <= wizard.currentStep ? styles.stepConnectorActive : ''
                }`}
              />
            )}
            <button
              className={`${styles.stepItem} ${
                idx === wizard.currentStep ? styles.stepActive : ''
              } ${
                idx < wizard.currentStep ? styles.stepCompleted : ''
              }`}
              onClick={() => wizard.goToStep(idx)}
            >
              <div className={styles.stepCircle}>
                {idx < wizard.currentStep ? (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                ) : (
                  idx + 1
                )}
              </div>
              <span className={styles.stepLabel}>{step.label}</span>
            </button>
          </React.Fragment>
        ))}
      </div>

      {/* Error alert */}
      {wizard.actionError && (
        <div className={styles.errorAlert}>
          <span>{wizard.actionError}</span>
          <button className={styles.errorDismiss} onClick={wizard.clearError}>
            &times;
          </button>
        </div>
      )}

      {/* Step content */}
      {wizard.currentStep === 0 && <BenchmarkStep wizard={wizard} />}
      {wizard.currentStep === 1 && <TrainStep wizard={wizard} />}
      {wizard.currentStep === 2 && <ConfigStep wizard={wizard} />}

      {/* Navigation */}
      <div className={styles.actions}>
        <div className={styles.actionsLeft}>
          {wizard.currentStep > 0 && (
            <button className={styles.btnSecondary} onClick={wizard.prevStep}>
              Back
            </button>
          )}
          <button className={styles.btnSecondary} onClick={wizard.reset}>
            Reset
          </button>
        </div>
        <div className={styles.actionsRight}>
          {wizard.currentStep < 2 && (
            <button
              className={styles.btnPrimary}
              onClick={wizard.nextStep}
              disabled={!wizard.canAdvanceToStep(wizard.currentStep + 1)}
            >
              Next Step
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

/* ============================================================
   Step 0: Benchmark
   ============================================================ */
interface StepProps {
  wizard: ReturnType<typeof useMLPipelineWizard>;
}

const BenchmarkStep: React.FC<StepProps> = ({ wizard }) => {
  const modelsInputRef = useRef<HTMLInputElement>(null);
  const queriesInputRef = useRef<HTMLInputElement>(null);

  const jobDone = wizard.benchmarkProgress.completed && wizard.benchmarkProgress.job?.status === 'completed';
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
        <button
          className={styles.btnSecondary}
          onClick={() => wizard.goToStep(1)}
        >
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
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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
            wizard.setBenchmarkConfig({ ...wizard.benchmarkConfig, concurrency: parseInt(e.target.value) || 4 })
          }
        />
        <div className={styles.formHint}>Number of concurrent requests during benchmark</div>
      </div>

      {/* Advanced Benchmark Settings */}
      <BenchmarkAdvancedSettings wizard={wizard} />

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
        <ProgressDisplay
          progress={wizard.benchmarkProgress.progress}
          job={wizard.benchmarkProgress.job}
          completed={wizard.benchmarkProgress.completed}
        />
      )}

      {/* Success */}
      {jobDone && (
        <div className={styles.successCard}>
          <div className={styles.successIcon}>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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
          <span>Benchmark failed: {wizard.benchmarkProgress.job?.error || 'Unknown error'}</span>
        </div>
      )}
    </div>
  );
};

/* ============================================================
   Step 1: Train
   ============================================================ */
const TrainStep: React.FC<StepProps> = ({ wizard }) => {
  const trainingDataInputRef = useRef<HTMLInputElement>(null);
  const jobDone = wizard.trainProgress.completed && wizard.trainProgress.job?.status === 'completed';
  const jobFailed = wizard.trainProgress.job?.status === 'failed';

  const algorithms: MLAlgorithm[] = ['knn', 'kmeans', 'svm', 'mlp'];
  const hasDataSource = wizard.benchmarkJobId !== null || wizard.trainingDataFile !== null;

  return (
    <div className={styles.stepContent}>
      <h2 className={styles.stepTitle}>Step 2: Train ML Classifiers</h2>
      <p className={styles.stepDescription}>
        Select the ML algorithms to train on your benchmark data. Each algorithm will produce a
        model that can be used for intelligent request routing.
      </p>

      {/* Training data source section */}
      {wizard.benchmarkJobId ? (
        <div className={styles.dataSourceBanner}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <polyline points="16 8 10 16 7 13" />
          </svg>
          <span>Using benchmark output from job <code>{wizard.benchmarkJobId}</code></span>
        </div>
      ) : (
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Training Data (JSONL)</label>
          <p className={styles.formHint} style={{ marginBottom: '0.5rem' }}>
            Upload your existing benchmark / training data file to skip the benchmark step.
          </p>
          <input
            ref={trainingDataInputRef}
            type="file"
            accept=".jsonl,.json"
            className={styles.fileInput}
            onChange={(e) => wizard.setTrainingDataFile(e.target.files?.[0] || null)}
          />
          <div
            className={`${styles.fileDropZone} ${wizard.trainingDataFile ? styles.fileDropZoneActive : ''}`}
            onClick={() => trainingDataInputRef.current?.click()}
          >
            {wizard.trainingDataFile ? (
              <div className={styles.fileDropName}>{wizard.trainingDataFile.name}</div>
            ) : (
              <>
                <div className={styles.fileDropIcon}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                </div>
                <div className={styles.fileDropText}>Click to upload training data JSONL</div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Algorithm selection */}
      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Algorithms</label>
        <div className={styles.algorithmGrid}>
          {algorithms.map((alg) => {
            const info = ML_ALGORITHM_INFO[alg];
            const selected = wizard.selectedAlgorithms.includes(alg);
            return (
              <div
                key={alg}
                className={`${styles.algorithmCard} ${selected ? styles.algorithmCardSelected : ''}`}
                onClick={() => wizard.toggleAlgorithm(alg)}
              >
                <div className={styles.algorithmHeader}>
                  <div className={styles.algorithmDot} style={{ background: info.color }} />
                  <span className={styles.algorithmName}>{info.label}</span>
                </div>
                <div className={styles.algorithmDesc}>{info.description}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Device selection — only relevant for MLP (PyTorch) */}
      {wizard.selectedAlgorithms.includes('mlp') && (
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Device</label>
          <div className={styles.deviceSelector}>
            <button
              className={`${styles.deviceOption} ${wizard.device === 'cpu' ? styles.deviceOptionSelected : ''}`}
              onClick={() => wizard.setDevice('cpu')}
            >
              CPU
            </button>
            <button
              className={`${styles.deviceOption} ${wizard.device === 'cuda' ? styles.deviceOptionSelected : ''}`}
              onClick={() => wizard.setDevice('cuda')}
            >
              CUDA (GPU)
            </button>
          </div>
        </div>
      )}

      {/* Advanced Training Settings */}
      <TrainAdvancedSettings wizard={wizard} />

      {/* Run button or progress */}
      {!wizard.trainJobId && (
        <button
          className={styles.btnPrimary}
          onClick={wizard.startTraining}
          disabled={wizard.actionLoading || wizard.selectedAlgorithms.length === 0 || !hasDataSource}
        >
          {wizard.actionLoading ? 'Starting...' : 'Train Models'}
        </button>
      )}

      {wizard.trainJobId && (
        <ProgressDisplay
          progress={wizard.trainProgress.progress}
          job={wizard.trainProgress.job}
          completed={wizard.trainProgress.completed}
        />
      )}

      {jobDone && (
        <div className={styles.successCard}>
          <div className={styles.successIcon}>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <polyline points="16 8 10 16 7 13" />
            </svg>
          </div>
          <div className={styles.successTitle}>Training Complete</div>
          <div className={styles.successMessage}>
            {wizard.trainProgress.job?.output_files?.length || 0} model(s) trained successfully.
            Proceed to generate deployment configuration, or train additional algorithms.
          </div>
          <button
            className={styles.btnSecondary}
            onClick={wizard.resetTraining}
            style={{ marginTop: '0.75rem' }}
          >
            Train Again (different algorithms)
          </button>
        </div>
      )}

      {jobFailed && (
        <div className={styles.errorAlert}>
          <span>Training failed: {wizard.trainProgress.job?.error || 'Unknown error'}</span>
          <button
            className={styles.btnSecondary}
            onClick={wizard.resetTraining}
            style={{ marginLeft: '0.75rem', padding: '0.25rem 0.75rem', fontSize: '0.75rem' }}
          >
            Retry
          </button>
        </div>
      )}
    </div>
  );
};

/* ============================================================
   Model Names Input — edits raw text, parses on blur
   ============================================================ */
const ModelNamesInput: React.FC<{
  value: string[];
  onChange: (names: string[]) => void;
  placeholder?: string;
}> = ({ value, onChange, placeholder }) => {
  const valueKey = value.join(',');
  const [raw, setRaw] = useState(value.join(', '));

  // Sync from parent when value changes externally (e.g. reset)
  useEffect(() => {
    setRaw(value.join(', '));
  }, [valueKey]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <input
      type="text"
      className={styles.textInput}
      value={raw}
      onChange={(e) => setRaw(e.target.value)}
      onBlur={() => {
        const parsed = raw.split(',').map((s) => s.trim()).filter(Boolean);
        onChange(parsed);
        setRaw(parsed.join(', '));
      }}
      placeholder={placeholder}
    />
  );
};

/* ============================================================
   Step 2: Config
   ============================================================ */
const ConfigStep: React.FC<StepProps> = ({ wizard }) => {
  const jobDone = wizard.configProgress.completed && wizard.configProgress.job?.status === 'completed';
  const jobFailed = wizard.configProgress.job?.status === 'failed';
  const algorithms: MLAlgorithm[] = ['knn', 'kmeans', 'svm', 'mlp'];

  return (
    <div className={styles.stepContent}>
      <h2 className={styles.stepTitle}>Step 3: Generate Configuration</h2>
      <p className={styles.stepDescription}>
        Define your routing decisions and model references. This will generate a deployment-ready
        YAML configuration file for semantic-router.
      </p>

      {/* Models path */}
      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Models Path</label>
        <input
          type="text"
          className={styles.textInput}
          value={wizard.modelsPath}
          onChange={(e) => wizard.setModelsPath(e.target.value)}
          placeholder="/tmp/ml-models"
        />
        <div className={styles.formHint}>Directory where trained ML model files will be stored</div>
      </div>

      {/* Decisions */}
      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Routing Decisions</label>
        <div className={styles.decisionList}>
          {wizard.decisions.map((dec, idx) => (
            <div key={idx} className={styles.decisionCard}>
              <div className={styles.decisionHeader}>
                <span className={styles.decisionNumber}>Decision #{idx + 1}</span>
                {wizard.decisions.length > 1 && (
                  <button className={styles.removeBtn} onClick={() => wizard.removeDecision(idx)}>
                    Remove
                  </button>
                )}
              </div>
              <div className={styles.decisionFields}>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Name</label>
                  <input
                    type="text"
                    className={styles.textInput}
                    value={dec.name}
                    onChange={(e) => wizard.updateDecision(idx, { name: e.target.value })}
                    placeholder="e.g. coding-router"
                  />
                </div>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Algorithm</label>
                  <select
                    className={styles.selectInput}
                    value={dec.algorithm}
                    onChange={(e) =>
                      wizard.updateDecision(idx, { algorithm: e.target.value as MLAlgorithm })
                    }
                  >
                    {algorithms.map((alg) => (
                      <option key={alg} value={alg}>
                        {ML_ALGORITHM_INFO[alg].label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Priority</label>
                  <input
                    type="number"
                    className={styles.textInput}
                    value={dec.priority}
                    onChange={(e) =>
                      wizard.updateDecision(idx, { priority: parseInt(e.target.value) || 0 })
                    }
                    min={0}
                    max={1000}
                    placeholder="100"
                  />
                </div>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Domains (comma-separated)</label>
                  <input
                    type="text"
                    className={styles.textInput}
                    value={dec.domains.join(', ')}
                    onChange={(e) =>
                      wizard.updateDecision(idx, {
                        domains: e.target.value.split(',').map((s) => s.trim()).filter(Boolean),
                      })
                    }
                    placeholder="e.g. coding, math, general"
                  />
                </div>
                <div className={styles.formGroup} style={{ gridColumn: '1 / -1' }}>
                  <label className={styles.formLabel}>Model Names (comma-separated)</label>
                  <ModelNamesInput
                    value={dec.model_names}
                    onChange={(names) => wizard.updateDecision(idx, { model_names: names })}
                    placeholder="e.g. gpt-4, codellama-34b"
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
        <button className={styles.addBtn} onClick={wizard.addDecision}>
          + Add Decision
        </button>
      </div>

      {/* Generate button or progress */}
      {!wizard.configJobId && (
        <button
          className={styles.btnSuccess}
          onClick={wizard.startConfigGeneration}
          disabled={wizard.actionLoading || wizard.decisions.length === 0}
        >
          {wizard.actionLoading ? 'Generating...' : 'Generate Config'}
        </button>
      )}

      {wizard.configJobId && (
        <ProgressDisplay
          progress={wizard.configProgress.progress}
          job={wizard.configProgress.job}
          completed={wizard.configProgress.completed}
        />
      )}

      {jobDone && wizard.configJobId && (
        <div className={styles.successCard}>
          <div className={styles.successIcon}>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <polyline points="16 8 10 16 7 13" />
            </svg>
          </div>
          <div className={styles.successTitle}>Configuration Generated</div>
          <div className={styles.successMessage}>
            Your deployment-ready configuration file is ready for download.
          </div>
          <a
            href={getDownloadUrl(wizard.configJobId)}
            className={styles.downloadBtn}
            download
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            Download YAML Config
          </a>
        </div>
      )}

      {jobFailed && (
        <div className={styles.errorAlert}>
          <span>Config generation failed: {wizard.configProgress.job?.error || 'Unknown error'}</span>
        </div>
      )}
    </div>
  );
};

/* ============================================================
   Advanced Settings: Benchmark
   ============================================================ */
const BenchmarkAdvancedSettings: React.FC<StepProps> = ({ wizard }) => {
  const [open, setOpen] = useState(false);

  return (
    <div className={styles.advancedSection}>
      <button className={styles.advancedToggle} onClick={() => setOpen(!open)}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
          style={{ transform: open ? 'rotate(90deg)' : 'rotate(0)', transition: 'transform 0.2s' }}>
          <polyline points="9 18 15 12 9 6" />
        </svg>
        Advanced Settings
      </button>
      {open && (
        <div className={styles.advancedContent}>
          <div className={styles.advancedGrid}>
            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Max Tokens</label>
              <input type="number" className={styles.numberInput} min={1} max={8192}
                value={wizard.benchmarkConfig.max_tokens ?? 1024}
                onChange={(e) => wizard.setBenchmarkConfig({ ...wizard.benchmarkConfig, max_tokens: parseInt(e.target.value) || 1024 })}
              />
              <div className={styles.formHint}>Maximum tokens in LLM response</div>
            </div>
            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Temperature</label>
              <input type="number" className={styles.numberInput} min={0} max={2} step={0.1}
                value={wizard.benchmarkConfig.temperature ?? 0}
                onChange={(e) => wizard.setBenchmarkConfig({ ...wizard.benchmarkConfig, temperature: parseFloat(e.target.value) || 0 })}
              />
              <div className={styles.formHint}>Generation temperature (0 = deterministic)</div>
            </div>
            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Query Limit</label>
              <input type="number" className={styles.numberInput} min={0}
                value={wizard.benchmarkConfig.limit ?? 0}
                onChange={(e) => wizard.setBenchmarkConfig({ ...wizard.benchmarkConfig, limit: parseInt(e.target.value) || 0 })}
              />
              <div className={styles.formHint}>Limit queries to process (0 = all)</div>
            </div>
            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Concise Prompts</label>
              <div className={styles.deviceSelector}>
                <button
                  className={`${styles.deviceOption} ${wizard.benchmarkConfig.concise ? styles.deviceOptionSelected : ''}`}
                  onClick={() => wizard.setBenchmarkConfig({ ...wizard.benchmarkConfig, concise: true })}
                >
                  On
                </button>
                <button
                  className={`${styles.deviceOption} ${!wizard.benchmarkConfig.concise ? styles.deviceOptionSelected : ''}`}
                  onClick={() => wizard.setBenchmarkConfig({ ...wizard.benchmarkConfig, concise: false })}
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
};

/* ============================================================
   Advanced Settings: Training
   ============================================================ */
const TrainAdvancedSettings: React.FC<StepProps> = ({ wizard }) => {
  const [open, setOpen] = useState(false);
  const showKnn = wizard.selectedAlgorithms.includes('knn');
  const showKmeans = wizard.selectedAlgorithms.includes('kmeans');
  const showSvm = wizard.selectedAlgorithms.includes('svm');
  const showMlp = wizard.selectedAlgorithms.includes('mlp');

  return (
    <div className={styles.advancedSection}>
      <button className={styles.advancedToggle} onClick={() => setOpen(!open)}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
          style={{ transform: open ? 'rotate(90deg)' : 'rotate(0)', transition: 'transform 0.2s' }}>
          <polyline points="9 18 15 12 9 6" />
        </svg>
        Advanced Settings
      </button>
      {open && (
        <div className={styles.advancedContent}>
          {/* General */}
          <div className={styles.advancedGroupTitle}>General</div>
          <div className={styles.advancedGrid}>
            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Quality Weight</label>
              <input type="number" className={styles.numberInput} min={0} max={1} step={0.05}
                value={wizard.qualityWeight}
                onChange={(e) => wizard.setQualityWeight(parseFloat(e.target.value) || 0.9)}
              />
              <div className={styles.formHint}>Weight for quality vs latency (0-1)</div>
            </div>
            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Batch Size</label>
              <input type="number" className={styles.numberInput} min={1} max={256}
                value={wizard.batchSize}
                onChange={(e) => wizard.setBatchSize(parseInt(e.target.value) || 32)}
              />
              <div className={styles.formHint}>Embedding generation batch size</div>
            </div>
          </div>

          {/* KNN */}
          {showKnn && (
            <>
              <div className={styles.advancedGroupTitle}>KNN Parameters</div>
              <div className={styles.advancedGrid}>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>K (neighbors)</label>
                  <input type="number" className={styles.numberInput} min={1} max={50}
                    value={wizard.knnK}
                    onChange={(e) => wizard.setKnnK(parseInt(e.target.value) || 5)}
                  />
                </div>
              </div>
            </>
          )}

          {/* KMeans */}
          {showKmeans && (
            <>
              <div className={styles.advancedGroupTitle}>KMeans Parameters</div>
              <div className={styles.advancedGrid}>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Clusters</label>
                  <input type="number" className={styles.numberInput} min={2} max={64}
                    value={wizard.kmeansClusters}
                    onChange={(e) => wizard.setKmeansClusters(parseInt(e.target.value) || 8)}
                  />
                </div>
              </div>
            </>
          )}

          {/* SVM */}
          {showSvm && (
            <>
              <div className={styles.advancedGroupTitle}>SVM Parameters</div>
              <div className={styles.advancedGrid}>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Kernel</label>
                  <select className={styles.selectInput} value={wizard.svmKernel}
                    onChange={(e) => wizard.setSvmKernel(e.target.value as SvmKernel)}>
                    <option value="rbf">RBF (Radial Basis Function)</option>
                    <option value="linear">Linear</option>
                  </select>
                </div>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Gamma</label>
                  <input type="number" className={styles.numberInput} min={0.001} max={100} step={0.1}
                    value={wizard.svmGamma}
                    onChange={(e) => wizard.setSvmGamma(parseFloat(e.target.value) || 1.0)}
                  />
                  <div className={styles.formHint}>RBF kernel parameter</div>
                </div>
              </div>
            </>
          )}

          {/* MLP */}
          {showMlp && (
            <>
              <div className={styles.advancedGroupTitle}>MLP Parameters</div>
              <div className={styles.advancedGrid}>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Hidden Sizes</label>
                  <input type="text" className={styles.textInput}
                    value={wizard.mlpHiddenSizes}
                    onChange={(e) => wizard.setMlpHiddenSizes(e.target.value)}
                    placeholder="256,128"
                  />
                  <div className={styles.formHint}>Comma-separated layer sizes</div>
                </div>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Epochs</label>
                  <input type="number" className={styles.numberInput} min={1} max={1000}
                    value={wizard.mlpEpochs}
                    onChange={(e) => wizard.setMlpEpochs(parseInt(e.target.value) || 100)}
                  />
                </div>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Learning Rate</label>
                  <input type="number" className={styles.numberInput} min={0.0001} max={1} step={0.0001}
                    value={wizard.mlpLearningRate}
                    onChange={(e) => wizard.setMlpLearningRate(parseFloat(e.target.value) || 0.001)}
                  />
                </div>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Dropout</label>
                  <input type="number" className={styles.numberInput} min={0} max={0.9} step={0.05}
                    value={wizard.mlpDropout}
                    onChange={(e) => wizard.setMlpDropout(parseFloat(e.target.value) || 0.1)}
                  />
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

/* ============================================================
   Progress Display Component
   ============================================================ */
interface ProgressDisplayProps {
  progress: ReturnType<typeof useMLPipelineWizard>['benchmarkProgress']['progress'];
  job: ReturnType<typeof useMLPipelineWizard>['benchmarkProgress']['job'];
  completed: boolean;
}

const ProgressDisplay: React.FC<ProgressDisplayProps> = ({ progress, job }) => {
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
};

export default MLSetupPage;
